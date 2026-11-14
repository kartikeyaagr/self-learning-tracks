---
id: "02.13"
course: "system-design-101"
module: "02"
slug: "load-shedding"
title: "Load shedding"
keyIdeas: ["Load shedding means rejecting some requests on purpose when overloaded, so the rest succeed.", "Without it, an overloaded server accepts everything, every request gets slow, clients time out and retry, and goodput drops toward zero.", "Reject early and cheaply (before doing expensive work), with a clear signal (HTTP 503 or 429) so clients back off.", "Prioritize: drop low-value traffic first (prefetch, analytics, batch jobs) and keep critical traffic (checkout, login).", "Signals for shedding: concurrency limits, queue length, CPU, or measured latency. Client-side throttling helps too."]
question: "Why is rejecting requests fast better than letting them all queue?"
answer: "If everything queues, every request waits longer until latency passes the client's timeout. The server then spends its capacity on requests whose clients have already given up and are retrying, so goodput drops towards zero while load keeps rising, and the system can stay stuck even after the spike passes. Rejecting excess requests fast is cheap, keeps the accepted ones within their deadlines so goodput stays at capacity, and tells clients to back off."
sources: [{"title": "AWS Builders' Library: Using load shedding to avoid overload", "url": "https://builder.aws.com/content/3Eun1EEyX6p2e3VYNyRLSJzLuMV/using-load-shedding-to-avoid-overload", "note": "Drop work early to protect goodput."}, {"title": "Google SRE book: Handling Overload", "url": "https://sre.google/sre-book/handling-overload/", "note": "Per-client limits, criticality, client-side throttling."}]
---

## Overload makes everyone fail

When a server receives more requests than it can handle, it doesn't just slow down a little. Requests queue for threads, CPU, memory, connections and locks, and past a certain point latency shoots up: the hockey stick from lesson 01.04, where a server at 95% utilisation has queues many times longer than one at 70%.

This turns into an **outage** because clients have timeouts. Once the server's latency passes the client's timeout, the client gives up and reports an error. But the server doesn't know that. It keeps working on the abandoned request, spends CPU and database time producing a response nobody will read, and only then moves on to the next one, which has probably also been abandoned. The server is 100% busy and produces almost nothing useful.

It helps to separate two measures:

- **Throughput**: requests the server handles per second.
- **Goodput**: requests handled **correctly and in time** for the client to use the answer.

If median latency equals the client timeout, half the requests fail even though the server is "working" on all of them. Then clients **retry**, adding even more load to a server that is already drowning. In a deep call chain where each layer retries, overload at the bottom amplifies itself (lesson 02.09), and the system can settle into a **stable overloaded state**: it stays down even after the original spike has passed, because retries alone keep it overloaded. This is sometimes called a **metastable failure**.

~~~dg-widget queue Latency against load for a single queue: flat at first, then a hockey stick as the server nears 100% busy.
~~~

## Shed load instead

**Load shedding** means that when a server is near its limit, it **rejects extra requests immediately**, with a quick error such as 503 Service Unavailable or 429 Too Many Requests, so that it can finish the requests it has accepted within their deadlines. Some users get a fast "try again", and everyone else gets a working service. Goodput stays flat as load rises past capacity, instead of collapsing.

~~~dg-compare The same server at twice its capacity. Without shedding every request slows past the timeout; with shedding, half are refused quickly and the other half stay fast.
[No shedding]
2x traffic -> Queue grows -> All requests time out
- Latency climbs past the client timeout
- Work done for clients who gave up
- Retries add even more load
- Goodput falls towards zero
[*Shedding]
2x traffic -(reject ~50% early, 503)-> Accepted requests stay fast
+ Goodput stays at capacity
+ Rejected clients learn quickly and back off
- Some users see errors
~~~

~~~dg-bars Useful responses per second for a server with capacity 1,000, at different offered loads. Without shedding, goodput collapses past capacity; with shedding it plateaus. (Illustrative numbers.)
1x load, no shedding | 1000 | 1,000/s
1.5x load, no shedding | 400 | ~400/s, most time out
2x load, no shedding | 50 | ~50/s
*2x load, with shedding | 980 | ~980/s
~~~

### How to decide when to shed

A server needs a signal that it is near its limit. Common choices:

- **Concurrency limit**: cap the number of requests in progress (for example 800). The next request beyond that is rejected. This is simple and directly tied to resource use; tools such as adaptive concurrency limiters adjust the cap automatically from observed latency.
- **Queue time**: reject a request if it has waited in the queue longer than some fraction of its deadline. A request that waited 900 ms of a 1 s budget is better dropped than served.
- **CPU or memory utilisation** above a threshold, measured over a short window.
- **Deadline already passed**: if the caller sent a deadline and it has expired, drop the request without doing any work.

### Make rejection cheap

Fast rejection only works if rejecting costs much less than serving. Check for hidden costs: a log line written for every rejection, a TLS handshake completed before rejecting, an expensive authentication step before the shedding check. Shed as early as possible, ideally before parsing the body. Also keep rejected requests **out of latency metrics**, or median latency will look great precisely while the service is shedding 60% of its traffic.

<div class="ex">
<div class="lab">Worked example</div>

**shedding with a concurrency limit**

A service handles each request in 50 ms of work, and can run 40 requests at once before latency starts climbing, so its capacity is about 40 / 0.05 = 800 requests per second. Clients time out after 1 s.

Without a limit, at 1,600 requests per second the queue grows by 800 every second. After 2 seconds, a new request waits behind 1,600 others, about 2 s, past the timeout. Practically every request now fails, and retries push the load higher.

With a limit of 40 in progress and a short queue of 40 (at most about 100 ms of waiting), requests beyond that get an immediate 503. About 800 per second are served in roughly 50 to 150 ms; about 800 per second are rejected in under a millisecond each and retry later with jitter. Half the users succeed at once, instead of almost none.

</div>

## Decide what to shed

Not all requests matter equally. Give each request a **criticality**. One widely used scheme has four levels:

~~~dg-layers Request criticality, from most to least protected. An overloaded server rejects the lowest level first.
CRITICAL_PLUS | must never be shed if avoidable, e.g. payment confirmation
*CRITICAL | default for user-facing traffic, e.g. login, checkout
SHEDDABLE_PLUS | can tolerate some failure, e.g. batch jobs that retry
SHEDDABLE | dropped first, e.g. prefetch, analytics, background sync
~~~

An overloaded server rejects the lowest criticality first, and only moves up a level when shedding the lower ones isn't enough. Criticality is **propagated** to downstream calls, so a backend knows that a request is part of a critical checkout, not a background job, even three hops away. Also:

- **Never shed health checks** from the load balancer. If the server rejects them, the balancer pulls it from the pool, its traffic moves to the others, and they overload in turn: shedding turns into a cascade.
- **Prefer real users over crawlers**, and callers within their quota over those bursting above it.
- **Let started work finish**: prioritise a request that completes a workflow (the payment step of a checkout already in progress) over one that starts a new one (a new visitor's home page). Abandoning half-finished work wastes everything already spent on it.

## Measure capacity correctly

Requests per second is a poor capacity measure, because requests vary wildly in cost (a cached profile read versus a full-text search), and the mix changes over the day. A limit of "1,000 requests per second" will be too loose when the mix shifts to expensive requests, and too tight when it shifts to cheap ones.

Measure capacity in **resources**, such as CPU seconds or database time consumed. Give each customer a **quota** in those units. Then, in a global overload, only the customers above their quota see errors, while customers within their share continue normally.

## Help clients back off

Shedding on the server still costs something per rejected request. Clients can help.

**Client-side adaptive throttling.** Each client tracks, over the last two minutes, how many requests it sent (`requests`) and how many the backend accepted (`accepts`). Normally the two are equal. When the backend starts rejecting, `requests` grows faster than `accepts`. Once `requests` exceeds `K × accepts` (Google uses K = 2), the client starts **rejecting some requests locally**, without sending them, with probability `max(0, (requests − K × accepts) / (requests + 1))`. The backend then spends no effort rejecting those requests at all, and the client automatically sends more again as the backend recovers.

**Retry budgets.** At most 3 attempts per request, and retries limited to **10% of a client's traffic**. Without the budget, a fully failing backend sees up to 3 times the normal request rate; with it, at most about 1.1 times. Retry only at the layer directly above the overloaded one, so retries don't multiply through the stack.

**Deadlines.** Pass the remaining time budget down the call chain. A server receiving a request whose deadline has passed drops it immediately, and a server whose remaining budget is too small for the work it would need to do can refuse up front.

~~~dg-seq Deadline propagation. Each hop passes on the time remaining, and a backend drops work whose caller has already given up.
Client > Frontend : GET /feed, deadline 1,000 ms
Frontend > Frontend : 300 ms queued under load
Frontend > Ranking : rank(user), 700 ms left
Ranking > Ranking : 720 ms queued
Ranking > Ranking : deadline passed: drop, no work done
Ranking -> Frontend : DEADLINE_EXCEEDED
Frontend -> Client : 503, Retry-After 2
~~~

<div class="ex">
<div class="lab">Worked example</div>

**adaptive throttling in numbers**

A client sends 1,000 requests per second to a backend that, overloaded, now accepts only 300 per second.

Over the window, requests = 1,000 and accepts = 300, so K × accepts = 600. The client rejects locally with probability (1,000 − 600) / 1,001 ≈ 40%.

It now sends about 600 per second, of which the backend accepts its 300. The backend rejects 300 instead of 700, saving that work. As the backend recovers and accepts rise, the local rejection probability falls back to zero on its own, with no configuration change.

</div>

## Queues and testing

- **Bound queues** by size and by time spent waiting, and drop requests that have waited too long.
- Under overload, serving **newest first** (LIFO) favours requests whose clients are still waiting, rather than ones that have probably timed out. Some systems switch from FIFO to LIFO only when the queue is long.
- **Load test past the breaking point.** Push to 2 or 3 times capacity and check that goodput rises to a plateau and stays flat, rather than falling off a cliff. Test recovery too: when load drops, does the service come back, or does it stay stuck because of retries?
- **Watch autoscaling.** If shedding keeps CPU looking normal, CPU-based autoscaling may never trigger. Scale on concurrency, queue depth, or the shed rate as well.

<div class="ex">
<div class="lab">Worked example</div>

**ticket sale**

A concert goes on sale and traffic jumps to 20 times normal. Without shedding, the ticket service's latency climbs past the 2 s client timeout within seconds; users hammer refresh, retries multiply the load, and almost nobody gets a ticket.

With shedding, each server accepts up to its tested limit of 800 concurrent requests and returns 503 with `Retry-After` for the rest. Browsing seat maps (SHEDDABLE) is rejected before seat holds and checkouts (CRITICAL), and a checkout already in progress is always allowed to finish.

Accepted checkouts complete in about 300 ms. Clients retry with jitter and within their retry budget, and a virtual waiting room at the edge admits users at the rate the backend can actually serve. The sale completes in 40 minutes with every ticket sold, instead of an hour-long outage.

</div>

## Common mistakes

- **No limit at all**, relying on timeouts to "naturally" cap load.
- **Shedding health checks**, causing cascades.
- **Expensive rejections**: heavy logging or full request parsing before deciding to reject.
- **Capacity limits in requests per second** when costs vary widely.
- **Unbounded retries** that keep the system overloaded after the spike ends.

## In an interview

When traffic can spike (sales, launches, breaking news), say how the service protects itself: "each server caps in-flight requests and returns 503 fast beyond that; requests carry a criticality so background work is shed first; clients retry with jitter within a 10% retry budget; deadlines propagate so abandoned work is dropped". Then mention load testing beyond capacity.

<div class="take">
<div class="lab">Key takeaway</div>

under overload, a fast "no" protects the requests you accept and keeps goodput at capacity instead of collapsing. Shed early and cheaply, by criticality, measure capacity in resources, propagate deadlines, and cap retries so clients don't finish the job of knocking you over.

</div>
