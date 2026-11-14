---
id: "16.01"
course: "system-design-101"
module: "16"
slug: "a-notification-system"
title: "A notification system"
keyIdeas: ["A notification system sends messages via push (APNs, FCM), SMS, email and in-app.", "Flow: services call a notification API → validate and check user preferences and rate limits → enqueue per channel → channel workers call third-party providers → record delivery status.", "Queues per channel isolate failures: an SMS provider outage doesn't block email.", "Avoid duplicates after crashes: attach a notification ID, and have workers check a dedup store before sending (at-least-once delivery with dedup).", "Templates, user opt-outs, quiet hours and retries with backoff are part of the core design."]
question: "How do you avoid sending the same notification twice after a worker crash?"
answer: "Give every notification a unique ID (or a deterministic key such as order-123-shipped) and keep a notification log. Before calling the provider, a worker atomically claims the row (update status from queued to sending where status is still queued); after the provider accepts, it marks it sent and only then acknowledges the queue message. If the worker crashes and the job is redelivered, the next worker's claim fails because the row is already sent or being sent, so it skips. A crash between the provider accepting and the log update can still cause a rare duplicate: decide per message type whether a stale sending row is resent (rare duplicate) or dropped (rare loss), and pass the notification ID to providers that deduplicate, such as an APNs collapse ID, to close most of the window."
sources: [{"title": "ByteByteGo: Design a Notification System", "url": "https://bytebytego.com/courses/system-design-interview/design-a-notification-system", "note": "Push, SMS, email providers and queues."}, {"title": "Hello Interview: Notification system breakdown", "url": "https://www.hellointerview.com/learn/system-design/problem-breakdowns/notification-system", "note": "Interview breakdown."}]
tags: ["Design"]
---

## What we are building

A notification system lets other services say "tell user 42 their order shipped" without knowing anything about push certificates, SMS carriers or email reputation. The order service, the payments service and the marketing team all call one internal API, and the notification system takes care of choosing the channel, respecting the user's wishes, talking to the outside providers and retrying when they fail.

It looks simple, and that is why it is a popular interview question. The hard parts are hidden: bursts that are 50 times the average load, third-party providers that throttle you or go down, at-least-once queues that deliver the same job twice, and users who will uninstall your app if they get the same message three times.

### Functional requirements

- Upstream services send a notification to a user by **push, email or SMS**, either immediately or at a **scheduled time**.
- Marketing can send a **campaign**: one message to a whole segment of users ("everyone in the UK who bought shoes last month").
- Users set **preferences**: which channels and categories they have opted out of, and **quiet hours** (no pushes between 22:00 and 07:00 local time).
- Messages are built from **templates** with variables (`"Your order {{id}} shipped"`), localised per user.
- Out of scope: in-app inbox, rich analytics dashboards, A/B testing of copy.

### Non-functional requirements

- **No lost notifications** once the API has accepted them. A password reset code that never arrives is a support ticket.
- **Very few duplicates.** We will see that "zero duplicates" is impossible with an outside provider, so the target is "rare and bounded".
- **Low latency for transactional messages**: a login code should reach the provider within about 5 seconds, even while a campaign is running.
- **Isolation**: a broken SMS provider must not slow down push or email.
- Burst tolerance, explained by the numbers below.

## Back-of-the-envelope numbers

<div class="ex">
<div class="lab">Worked example</div>

**sizing the load**

Normal traffic is about 10 million notifications a day. 10,000,000 / 86,400 seconds is roughly 115 per second, call it 100 per second on average.

A campaign might send 1 million messages and marketing wants it delivered within 5 minutes: 1,000,000 / 300 is about 3,300 per second, on top of normal traffic.

Design for bursts of about 5,000 per second, roughly 50 times the average. That ratio is the key fact: you cannot size providers for the peak (they rate-limit you), so you need a buffer that absorbs the burst and drains it at a safe rate.

Storage for the notification log: each row (IDs, user, channel, template, status, timestamps) is about 500 bytes. 10 million a day is 5 GB a day, about 1.8 TB a year. Keep 30 to 90 days hot and archive the rest.

</div>

The ratio of peak to average tells you most of the architecture. If the load were flat at 100 per second, a single service calling providers synchronously would work. At 50 times bursts with providers that throttle, you need queues.

## The providers

You never send to phones directly. Each channel goes through a provider, and each provider has its own identifier, limits and failure modes:

- **iOS push** through Apple Push Notification service (APNs). The app registers with Apple and receives a **device token**, an opaque string that identifies this app on this device. Your server sends the token plus a payload (up to 4 KB) to APNs over HTTP/2.
- **Android push** through Firebase Cloud Messaging (FCM), also by device token (called a registration token).
- **SMS** through a provider such as Twilio or Vonage, to a phone number. SMS costs real money per message (often a few cents), so a duplicate SMS is a direct cost.
- **Email** through a provider such as SendGrid, Mailgun or Amazon SES. Email providers care about your sending reputation: bounces and spam complaints can get you throttled.

So the system must store **contact info** per user: a list of device tokens (a user may have a phone and a tablet), a verified phone number and an email address. These are collected when the app is installed, when the user signs up, or when the app starts and refreshes its token. A device token can change at any time (reinstall, restore from backup), so the app sends its current token on each launch and the server upserts it.

~~~dg-tree The data the notification system keeps, grouped by what it is used for.
Notification data
  Users
    Contact info | tokens, phone, email
    Preferences | opt-outs, quiet hours
  Templates | per name and locale
  Notification log | one row per message
    Status and attempts
    Provider message ID
  Schedules | due time index
~~~

## The API

Upstream services call one endpoint. Keep it small:

```
POST /v1/notifications
{ "idempotency_key": "order-123-shipped",
  "user_id": 42,
  "template": "order_shipped",
  "params": { "id": "123" },
  "channels": ["push", "email"],
  "priority": "transactional",
  "send_at": null }
-> 202 Accepted { "notification_id": "n-789" }
```

The response is **202 Accepted**, not 200, because the notification has been accepted for delivery, not delivered. The caller should not wait for APNs. The `idempotency_key` lets the caller retry safely: if the order service times out and calls again with the same key, the API returns the same `notification_id` instead of creating a second notification. A deterministic key built from business facts (`order-123-shipped`) is better than a random one because even a restarted caller produces the same key.

Campaigns use a separate endpoint, `POST /v1/campaigns {segment_id, template, send_at}`, because they behave differently: one request becomes a million jobs.

## The high-level design

~~~dg-flow Notification pipeline: accept fast, queue per channel and priority, let workers talk to providers.
[Accept] Order service -(send user 42)-> *Notification API -(prefs, template)-> Push queue
[Deliver] Push queue -(workers)-> Push worker -(device token)-> APNs or FCM
[Other channels] Notification API -(email job)-> Email queue -(workers)-> Email provider
~~~

1. **Notification API.** Authenticates the calling service, validates the request, checks the idempotency key, looks up the user's contact info and **preferences** (drop the channel if the user opted out; delay it if it is quiet hours in the user's time zone), renders the **template** in the user's language, writes a row to the notification log with status `queued`, and puts one job per channel on a queue.
2. **One queue per channel.** If the SMS provider is down, SMS jobs pile up in their own queue while push and email carry on. This is the bulkhead idea from resilience engineering: a failure fills one compartment, not the whole ship. Queues also absorb campaign bursts: the API accepts a million jobs in seconds, and workers drain them at whatever rate the providers allow.
3. **Workers.** Each worker takes a job, claims it in the log, calls the provider, records the result and acknowledges the job. Workers are stateless, so you scale them by adding more.
4. **Notification log.** A table with one row per notification and channel: status (`queued`, `sending`, `sent`, `failed`, `retrying`), attempt count, provider message ID and timestamps. It powers retries, deduplication, support questions ("did the user get the reset email?") and delivery analytics. A wide-column store such as Cassandra or a sharded relational table keyed by notification ID both work; queries are almost always by ID or by user.

### Scheduled notifications

A notification with `send_at` in the future goes into a **schedule table** indexed by due time (or a delay queue). A scheduler process polls every second for rows that are due (`WHERE send_at <= now() AND state = 'pending' LIMIT 1000`), moves them onto the channel queues and marks them dispatched. Run several schedulers safely by having each claim rows with `SELECT ... FOR UPDATE SKIP LOCKED`, so two schedulers never take the same row. Quiet hours reuse the same mechanism: a push that would arrive at 02:00 local time is rescheduled for 07:00.

### Campaigns

A campaign is expanded by a **fan-out job** that reads the segment in pages of, say, 10,000 users and enqueues per-user jobs. It writes progress (last user ID processed) so it can resume after a crash rather than starting again and sending duplicates. Crucially, campaign jobs go to **bulk queues**, never to the transactional ones.

### Priority

How do you deliver a login code within 5 seconds while a 1-million-message campaign is draining? If both share one FIFO queue, the code waits behind up to a million marketing messages, which at 3,000 per second is over five minutes. The answer is **separate queues and worker pools per priority**: `push-transactional`, `push-bulk`, and so on. Transactional workers are sized for the normal rate with headroom and never touch bulk jobs. You can also reserve a share of each provider's rate limit for transactional traffic so a campaign cannot starve it at the provider either.

~~~dg-compare One shared queue versus queues split by channel and priority.
[One shared queue]
API -> Single queue -> Workers
- Login code waits behind campaign
- SMS outage blocks push and email
+ Simple to run
[*Split by channel and priority]
API -> Push transactional -> Fast pool
+ Codes skip the marketing backlog
+ Provider outage stays contained
- More queues to monitor
~~~

## No duplicates after a worker crash

This is the lesson's central question. Queues such as SQS, RabbitMQ and Kafka give **at-least-once** delivery. A worker receives a job, does the work, then acknowledges it. If the worker crashes after doing the work but before acknowledging, the queue assumes the job was never processed and delivers it again (after a visibility timeout in SQS, or on consumer group rebalance in Kafka). Another worker sends the push again. The user gets two "your order shipped" messages.

You cannot fix this inside the queue. The fix is to make sending **idempotent**: doing it twice has the same effect as doing it once. You do that with a **deduplication key** and a claim step:

- Each notification has a unique **notification ID**, created by the API (and derived from the caller's idempotency key, so upstream retries also map to the same ID).
- Before sending, the worker **claims** the row with an atomic conditional update: `UPDATE log SET status='sending', claimed_at=now(), worker='B' WHERE id='n-789' AND status IN ('queued','retrying')`. If one row changed, this worker owns the send. If zero rows changed, someone else has already sent it or is sending it, so the worker skips and acknowledges.
- After the provider accepts, the worker sets `status='sent'` and stores the provider's message ID. Only then does it acknowledge the queue message.

~~~dg-seq A redelivered job finds the notification already sent and skips it.
Worker A > Log : claim n-789 (queued to sending) OK
Worker A > APNs : send push
APNs -> Worker A : accepted
Worker A > Log : n-789 = sent
Worker A > Worker A : crash before ack
Worker B > Log : claim n-789: already sent, skip and ack
~~~

### The window that remains

There is still a gap: the worker crashes **after** the provider accepted the message but **before** writing `sent`. The row says `sending` with a `claimed_at` from a minute ago. What should the next worker do? It cannot know whether the provider received the message. You have three options:

- **Treat stale `sending` as unsent** after a timeout (say 60 seconds) and resend. Rare duplicate, never a lost message. Right for login codes and order updates.
- **Treat stale `sending` as sent** and give up. Never a duplicate, rare lost message. Right for low-value marketing.
- **Let the provider deduplicate.** Some providers accept your own ID: APNs has an `apns-collapse-id` header (the device shows only the latest message with that ID), FCM has collapse keys, and many email and SMS APIs accept an idempotency key or custom message ID. Passing the notification ID closes most of the window.

Exactly-once delivery across a third party you do not control is impossible in general (it is the two generals problem: you can never be sure your last message arrived). The engineering goal is to make the duplicate window small, rare and chosen on purpose per notification type.

<div class="ex">
<div class="lab">Worked example</div>

**tracing a crash with numbers**

At 12:00:00.000 worker A claims n-789 and calls APNs. At 12:00:00.080 APNs returns 200. At 12:00:00.081 worker A's container is killed by a deploy, before its log write.

The SQS visibility timeout is 30 seconds, so at 12:00:30 the job reappears and worker B picks it up. Its claim fails because the row is `sending`, not `queued`. Worker B checks `claimed_at`: 30 seconds old, below the 60-second stale threshold, so it releases the message back with a delay.

At 12:01:05 the job returns, the claim is now 65 seconds stale, the template is "order shipped" (resend allowed), so worker B takes over and sends again with the same `apns-collapse-id`. The phone shows one notification, replaced in place. Without the collapse ID, the user sees two, once, and the metric `stale_reclaims` goes up by one.

</div>

## Getting it in without losing it

Duplicates happen on the way out; losses happen on the way in. The API must write the log row **and** put the job on the queue. If it writes the row and then crashes before enqueueing, the notification is stuck at `queued` forever. If it enqueues first and the database write fails, a worker finds no row.

The standard fix is the **transactional outbox**: the API writes the log row and an outbox row in one database transaction, and a separate relay reads the outbox and publishes to the queue, deleting outbox rows only after the queue confirms. A sweeper that finds rows stuck in `queued` for more than a few minutes and re-enqueues them is a cheap second line of defence. Because sending is idempotent, re-enqueueing too eagerly is harmless.

## Retries, backoff and dead letters

Provider calls fail in two ways, and they need different handling:

- **Transient errors**: timeouts, HTTP 429 (too many requests), HTTP 5xx. Retry with **exponential backoff and jitter**: wait about 1 s, 2 s, 4 s, 8 s, each multiplied by a random factor between 0.5 and 1.5 so that 50,000 failed jobs do not all retry in the same millisecond. Honour a `Retry-After` header when the provider sends one.
- **Permanent errors**: invalid device token, unsubscribed email, malformed phone number. Do not retry. APNs returns 410 Unregistered and FCM returns UNREGISTERED for tokens that no longer exist: delete those tokens from contact info so you stop paying to fail.

After a limit (say 6 attempts, about a minute of total backoff for transactional, several hours for bulk), move the job to a **dead-letter queue** and mark the row `failed`. Someone (or an automated replay tool) looks at the dead-letter queue after the provider recovers.

~~~dg-state The life of one notification in the log. Workers move it forward only with conditional updates.
[*] -> Queued
Queued -> Sending : worker claims
Sending -> Sent : provider accepts
Sending -> Retrying : transient error
Retrying -> Sending : backoff elapsed
Retrying -> *Failed : attempts exhausted
Sending -> Queued : stale claim reclaimed
~~~

### Circuit breakers per provider

If SendGrid starts timing out on every call, workers that keep calling it waste time and pile up connections. A **circuit breaker** per provider (lesson 06.10) opens after, say, 50% errors over 30 seconds, and workers then return jobs to the queue with a delay instead of calling. If you have a second provider for the channel (many teams keep a backup SMS provider), the breaker opening is the trigger to fail over.

## Rate limits in both directions

- **Provider limits.** Providers throttle you. Use a token bucket per provider account, shared by all workers through Redis, so the fleet as a whole stays under the limit rather than each worker guessing.
- **Per-user limits.** Don't send someone 50 pushes an hour because five services each think their event is important. A per-user counter (for example, at most 5 marketing pushes a day and 20 notifications an hour in total) protects the user and your uninstall rate. Transactional messages such as security codes usually bypass the marketing cap.

<div class="ex">
<div class="lab">Worked example</div>

**draining a campaign under a provider limit**

The campaign has 1,000,000 push jobs. FCM and APNs together allow you 2,000 bulk messages per second (you keep another 500 per second reserved for transactional).

Drain time is 1,000,000 / 2,000 = 500 seconds, about 8.5 minutes. Marketing asked for 5 minutes, so either negotiate a higher limit, accept 8.5 minutes, or start the campaign earlier. Queue depth peaks at nearly 1 million, so the queue must hold at least that many jobs of about 1 KB each: 1 GB, easily within SQS or Kafka limits.

Meanwhile a login code arrives on `push-transactional`, where the queue depth is near zero, and is sent in well under a second.

</div>

## Monitoring

- **Queue depth and age of the oldest message per queue** are the key signals. A growing transactional queue means workers or a provider cannot keep up; page someone.
- **Provider error rate and latency**, per provider.
- **Delivery funnel per template**: accepted, sent, delivered (where providers report it), opened. A sudden drop in opens for one template often means a broken link or a spam-folder problem.
- **Stale reclaims and dead-letter counts**: these are your duplicate and loss indicators.

## Common mistakes

- **Calling providers synchronously from the API.** The caller now waits on APNs, and a campaign takes the API down.
- **One queue for everything.** Login codes wait behind marketing and one bad provider stalls all channels.
- **Acknowledging before sending.** That turns at-least-once into at-most-once: a crash loses the message.
- **Claiming "exactly once".** Say "effectively once, with a small window we choose per message type".
- **Retrying permanent errors.** Invalid tokens and bounced addresses should be removed, not retried.
- **Ignoring time zones.** Quiet hours in the server's time zone wake people up.

## In an interview

Start with the burst ratio, because it justifies the queues. Draw the API, per-channel queues and workers early, then spend your depth on the duplicate problem: at-least-once delivery, the claim step, the remaining window and provider-side deduplication. Mention priority queues before the interviewer asks "what about the login code during a campaign?". Good questions to ask: which notifications must never be duplicated (SMS costs money), whether marketing has a delivery deadline, and whether a second provider per channel is available.

<div class="take">
<div class="lab">Key takeaway</div>

accept notifications through one API that checks preferences and renders templates, then use one queue per channel and priority with workers calling APNs, FCM, SMS and email providers. Queues deliver at least once, so give each notification an ID, claim it atomically in a notification log before sending, retry with backoff, dead-letter what keeps failing, and pass your ID to providers that can deduplicate.

</div>
