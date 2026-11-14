---
id: "14.12"
course: "system-design-101"
module: "14"
slug: "sliding-window-rate-limiter"
title: "Sliding window rate limiter"
keyIdeas: ["**Fixed window**: count requests per calendar minute. Simple, but a burst at 0:59 and another at 1:00 lets 2x the limit through within a few seconds.", "**Sliding window log**: store each request timestamp, count those in the last 60 s. Exact, but uses memory per request.", "**Sliding window counter**: estimate = current window count + previous window count × fraction of the previous window still in range. Cheap and close enough (Cloudflare uses this).", "**Token bucket**: tokens refill at a steady rate up to a burst size; each request spends one. Allows controlled bursts.", "Store counters in Redis with atomic operations (INCR or Lua scripts) so all servers share them."]
question: "Why does a fixed window allow 2x the limit at a window boundary?"
answer: "A fixed window resets its counter at fixed calendar times, so it only limits requests inside each calendar window, not inside every rolling period of the same length. A client can use the whole limit at the very end of one window and the whole limit again at the start of the next, because the counter resets in between. For example, with 100 per minute, 100 requests at 12:00:59 and 100 at 12:01:00 are both allowed, giving 200 in about a second: twice the limit for any 60-second span that crosses the boundary. A sliding window counter fixes this by estimating 100 × 60/60 + 0 = 100 at 12:01:00 and rejecting the second burst."
sources: [{"title": "ByteByteGo: Design a Rate Limiter", "url": "https://bytebytego.com/courses/system-design-interview/design-a-rate-limiter", "note": "Token bucket, sliding window log, sliding window counter."}, {"title": "Cloudflare: Counting things, a lot of different things", "url": "https://blog.cloudflare.com/counting-things-a-lot-of-different-things/", "note": "Sliding window at Cloudflare scale."}]
tags: ["Code"]
---

## What the algorithm has to answer

Every rate limiter, wherever it runs (lesson 14.11), has to answer one question on every request: "has this client used up its allowance for the current period?" The algorithm decides what "the current period" means and how much state you keep to answer it. The choices trade **accuracy** (does it ever allow more than the limit, or block someone under it?), **memory per client**, **cost per request** and **burst behaviour** (can a client send its whole allowance in one second?).

There are four common algorithms. This lesson builds them up from the simplest, shows exactly where each one leaks, and works through the sliding window counter that large edge networks use.

## Fixed window: simple but leaky

A **fixed window counter** keeps one counter per client per calendar window, keyed like `count:user42:12:01`. Each request increments it; if the result is above the limit, reject. When the minute changes, a new key starts at zero. In Redis it is just two commands: `INCR key` and, on the first request of the window, `EXPIRE key 60`.

It is cheap (one integer per client) and easy to reason about. The flaw is at the **boundary**. With a limit of 100 per minute, a client can send 100 requests at 12:00:59 (allowed: window 12:00 has room) and 100 more at 12:01:00 (allowed: window 12:01 just reset). That is **200 requests in about one second**, twice the limit, and it can happen at every minute boundary.

Why exactly 2×? The limiter only ever looks at calendar windows, but a real 60-second period can start anywhere. Any 60-second span that straddles a boundary covers the end of one window and the start of the next, and each part can hold up to the full limit. So the worst case for any rolling 60 seconds is 2 × limit. Another example: a limit of 5 per minute lets 10 requests through between 2:00:30 and 2:01:30 if 5 arrive at 2:00:59 and 5 at 2:01:00.

There is a second, subtler problem: **synchronised clients**. If many clients are blocked until the window resets, they all retry at 12:01:00, creating a spike at every boundary. Adding random jitter to `Retry-After` helps.

## Sliding window log: exact but heavy

To be exact, store the **timestamp of every accepted request**. In Redis, use a sorted set per client with the timestamp as the score. On each request:

1. `ZREMRANGEBYSCORE key 0 (now − 60s)` removes timestamps older than one window.
2. `ZCARD key` counts what is left.
3. If the count is under the limit, `ZADD key now now` and allow; otherwise reject.

Run it as a Lua script so the three steps are atomic. This is exact for any rolling 60 seconds: no boundary burst, ever.

The cost is memory and work. A limit of 5,000 per hour means up to 5,000 timestamps per client, each about 16 to 20 bytes in a sorted set with overhead, so roughly 100 KB per heavy client. For a million clients near their limit, that is around 100 GB. Each request also does a logarithmic-time sorted set update. Some implementations store rejected requests' timestamps too (so a client hammering the API stays blocked), which makes it heavier still.

## Sliding window counter: the approximation

The **sliding window counter** gets most of the log's accuracy for the fixed window's cost. Keep **two counters** per client: the previous window's count and the current window's count. Estimate the count over the last full window by assuming the previous window's requests were spread evenly across it:

`rate = prev_count × (window − elapsed) / window + curr_count`

Here `elapsed` is how far you are into the current window. The fraction `(window − elapsed) / window` is how much of the previous window still lies inside the rolling 60 seconds that ends now.

<div class="ex">
<div class="lab">Worked example</div>

**one decision, worked through**

Limit 50 per minute. It is 12:01:15, so we are 15 seconds into the current minute, with 18 requests so far. The previous minute (12:00) had 42.

The rolling 60 seconds ending now runs from 12:00:15 to 12:01:15. It covers the last 45 seconds of the previous minute (45/60 = 75% of it) plus the current 15 seconds.

`rate = 42 × 45/60 + 18 = 31.5 + 18 = 49.5`. That is under 50, so the request is allowed, and the current counter becomes 19.

Five seconds later (elapsed 20 s, assume no more requests), `rate = 42 × 40/60 + 19 = 28 + 19 = 47`. The estimate falls as the old window slides out of view, freeing allowance gradually rather than all at once.

</div>

~~~dg-compare The same boundary attack against a fixed window and a sliding window counter.
[Fixed window]
12:00:59 -(100 requests, count 100)-> allowed
12:01:00 -(counter resets to 0)-> 100 more allowed
- 200 requests in 1 second
- worst case 2x the limit
[*Sliding window counter]
12:00:59 -(100 requests, count 100)-> allowed
12:01:00 -(estimate 100 x 60/60 + 0 = 100)-> rejected
+ second burst blocked
+ still only two integers per client
~~~

Applying it to the boundary attack: at 12:01:00 with 100 requests in the previous minute, `100 × 60/60 + 0 = 100`, so the second burst is **rejected**. As time passes the allowance comes back smoothly: at 12:01:30 the estimate is `100 × 30/60 = 50`, so 50 more requests are allowed. The 2× burst is gone.

### How wrong can the approximation be?

The even-spread assumption is the only source of error. If the previous window's 100 requests all came in its first second, by 12:01:30 they are truly outside the rolling window, yet the estimate still counts 50 of them, so the limiter is stricter than necessary. If they all came in its last second, the estimate undercounts and allows slightly more than an exact log would. The error is bounded by the previous window's count times the fraction remaining, and in practice it averages out.

Cloudflare measured it on 400 million requests from 270,000 sources. Only **0.003%** of requests were wrongly allowed or blocked, and the average difference from the true rate was 6%. There were no false positives (nobody under the limit was blocked); 3 sources slipped through, each by less than 15% over the limit.

Why Cloudflare chose it:

- **Two numbers per counter**, so memory stays tiny across millions of rules and clients.
- **One atomic operation per request**: a memcache `INCR` on the current counter, plus one `GET` that fetches both counters. Cloudflare's stack could rely only on `GET`, `SET` and `INCR`; compare-and-swap failed under attack because of contention on hot keys.
- **One setting**: just a rate. The leaky bucket used by NGINX needs a rate and a burst size, which are hard for customers to tune correctly.

## How it runs at the edge

- **Anycast** routing sends a client to its nearest data centre (a point of presence, or PoP), and a given IP normally keeps reaching the same one. So each PoP counts **on its own**, with no global coordination: lower latency, and one PoP failing does not affect others. The cost is that a client deliberately spread across PoPs could get a limit per PoP, which is acceptable for abuse protection.
- Inside a PoP, counters live in a **memcache cluster behind Twemproxy**, sharded with consistent hashing (lesson 04.07).
- **Counting is asynchronous**: increments happen off the request path, so legitimate requests are not slowed.
- When a client goes over, a **mitigation flag** with an end time is set. Servers check only the flag and cache it locally until it expires, so an attacking source costs almost nothing per request after the first few.

~~~dg-seq At the edge: counting happens off the hot path, and once a client is over, a cached flag blocks it cheaply.
Client > Edge server : request 1 from IP X
Edge server > Edge server : no local block flag, serve request
Edge server > Memcache : async INCR current window for X
Edge server > Memcache : GET prev and current counts for X
Memcache -> Edge server : estimate 1,040 of limit 1,000
Edge server > Memcache : SET block flag for X until 12:03:00
Client > Edge server : request 2 from IP X
Edge server > Edge server : local flag cached, return 429 at once
~~~

This handled several billion requests a day and attacks of up to 400,000 requests per second against a single domain.

## The token bucket, for comparison

The **token bucket** is the other common choice. A bucket holds up to *capacity* tokens and refills at *rate* tokens per second; each request takes one token, and a request with no token is rejected. You store only two numbers per client, `tokens` and `last_refill`, and compute the refill lazily on each request: `tokens = min(capacity, tokens + (now − last_refill) × rate)`.

It deliberately **allows bursts** up to capacity and then enforces the average rate. For example, capacity 4 and refill 2 per second: an idle client can send 4 requests at once, then 2 per second after that. Choose a sliding window when you want "never more than N in any 60 seconds", and a token bucket when short bursts are fine or even desirable (Stripe allows bursts so customers can handle flash sales). The **leaky bucket** is its mirror image: requests enter a queue that drains at a fixed rate, smoothing output but adding delay.

~~~dg-widget bucket A token bucket in action: bursts pass while tokens last, then requests are rejected until it refills at a steady rate.
~~~

<div class="ex">
<div class="lab">Worked example</div>

**memory for 10 million clients**

Fixed window or sliding counter: two 8-byte integers plus key overhead, about 80 to 100 bytes per client in Redis: roughly 1 GB for 10 million clients.

Token bucket: the same, about 1 GB.

Sliding log at a limit of 1,000 per hour, with an average client holding 200 timestamps at about 20 bytes each: 4 KB per client, 40 GB in total, and heavy clients at the limit cost 20 KB each. The log is 40 times larger for the same clients, which is why it is reserved for low limits where exactness matters (5 password attempts per 15 minutes).

</div>

~~~dg-matrix The four algorithms compared: the sliding window counter and token bucket are the usual production choices.
Algorithm | Memory per client | Boundary burst | Allows deliberate bursts | Exact
Fixed window | 1 counter | yes, up to 2x limit | partial, by accident | no
Sliding log | 1 timestamp per request | no | no | yes
Sliding counter | 2 counters | no, nearly none | no | partial, 0.003% wrong
Token bucket | tokens + last refill | no | yes, up to capacity | yes, for its own definition
~~~

## Common mistakes

- **Using a fixed window for security limits** where a 2× burst matters (login attempts).
- **Non-atomic updates**, which let concurrent requests slip past any of these algorithms.
- **Forgetting TTLs**, so idle clients' counters fill memory forever.
- **A sliding log for high limits**, burning gigabytes to be exact where an approximation would do.
- **Clock disagreement** between servers computing windows; use the store's clock (Redis `TIME`) or a single time source.

## In an interview

If asked to design a rate limiter, name the algorithms, explain the fixed window's boundary problem in one sentence with numbers, and pick either a sliding window counter (smooth, cheap, nearly exact) or a token bucket (allows bursts) with a reason. Show the formula and one worked decision. That level of detail is usually what distinguishes a strong answer.

<div class="take">
<div class="lab">Key takeaway</div>

fixed windows reset at boundaries and let up to 2× the limit through any rolling period. A sliding log is exact but stores every timestamp. The sliding window counter keeps two numbers and weights the previous window by its overlap; at large scale it was wrong on only 0.003% of requests. Use a token bucket when you want controlled bursts.

</div>
