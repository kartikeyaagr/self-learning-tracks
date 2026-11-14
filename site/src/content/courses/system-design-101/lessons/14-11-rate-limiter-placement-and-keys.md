---
id: "14.11"
course: "system-design-101"
module: "14"
slug: "rate-limiter-placement-and-keys"
title: "Rate limiter placement and keys"
keyIdeas: ["Where to rate limit: at the edge (CDN/gateway) to stop floods cheaply, and in services for per-feature limits.", "What to key on: **IP** for anonymous traffic (but NAT and proxies share IPs, and attackers rotate them), **user ID** for logged-in actions, **API key** for developer quotas and billing tiers. Often combine several.", "Return HTTP 429 with a `Retry-After` header and rate-limit headers so clients can back off.", "Stripe uses several limiters: request rate, concurrent requests, fleet-wide load shedding by priority.", "Decide fail-open vs fail-closed if the rate limiter's store is down."]
question: "Rate limit by IP, user ID or API key? When each?"
answer: "Use **user ID** for authenticated traffic: it is fair per person and follows them across devices and IP addresses. Use **API key** for developer or B2B APIs, where limits match the customer's plan and billing and cover all of that customer's servers. Use **IP** for unauthenticated endpoints such as login, signup and public pages, and as a first line against floods, with higher limits because NAT puts many real users behind one IP (and by /64 prefix for IPv6). For login, also limit per target account so attacks spread across many IPs are caught. In practice combine them, with the most restrictive applicable rule winning."
sources: [{"title": "Stripe: Scaling your API with rate limiters", "url": "https://stripe.com/blog/rate-limiters", "note": "4 kinds of limiters."}, {"title": "Hello Interview: Rate limiter breakdown", "url": "https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-rate-limiter", "note": "Placement and keys."}]
---

## What a rate limiter is for

A **rate limiter** caps how many requests a client may make in a period: 100 per second per user, 5 login attempts per minute per account, 10,000 calls per day per API key. Requests over the limit are rejected straight away, usually with HTTP **429 Too Many Requests**. It protects the system from a single client using more than its share, whether by accident (a retry loop, a runaway test script, a buggy mobile release) or on purpose (scraping, credential stuffing, denial of service). It also enforces commercial plans: the free tier gets 60 requests a minute, the paid tier 6,000.

This lesson is about **where** to put the limiter, **what key** to count by and how to make shared counting correct and fast. Lesson 14.12 covers the counting algorithms themselves, and lesson 15.02 puts it all together in a design.

## Rate limiters and load shedders

Two related tools solve different problems:

- A **rate limiter** limits how fast each **client** can send. It is about fairness and suits cases where a client can spread its requests out without changing the outcome.
- A **load shedder** decides based on the **whole system's** health. During overload it drops low-priority traffic so critical traffic survives, regardless of which client sent it (lesson 02.13).

Stripe runs four, in order of how often they fire:

1. **Request rate limiter**: N requests per second per user, with short bursts allowed. It is the most important; it rejected millions of requests in one month, mostly from runaway test scripts. Test mode and live mode get the same limits, so scripts do not break on launch.
2. **Concurrent requests limiter**: at most, say, 20 requests **in flight** per user. This protects expensive endpoints from clients that retry slow requests and pile on. It fired about 12,000 times in a month.
3. **Fleet usage load shedder**: requests are split into critical (creating a charge) and non-critical (listing charges). A share of capacity, say 20%, is always kept for critical requests; non-critical ones beyond their share get HTTP 503.
4. **Worker utilisation load shedder**: the last line of defence. When workers run out, it sheds traffic in order: test mode first, then GETs, then POSTs, keeping critical methods longest. It sheds and restores gradually to avoid flapping. It fired about 100 times in a month.

~~~dg-bars How often each of Stripe's limiters fired in one month: everyday per-user limits do most of the work, emergency shedding is rare.
*Request rate limiter | 3000000 | millions
Concurrent requests limiter | 12000 | about 12,000
Worker utilisation shedder | 100 | about 100
~~~

Note the second one: a concurrency limit catches a problem a rate limit misses. A client sending 10 requests per second is well under a 100 per second limit, but if each request takes 30 s because the endpoint is slow, that client holds 300 workers. Counting requests **in flight** (increment on start, decrement on finish) bounds that.

### Rollout advice

- **Fail open** by default: if the limiter's store is down or the limiter has a bug, let traffic through rather than taking the API down.
- Return **429** for rate limits and **503** for load shedding, with messages that tell the client what to do.
- Put each limiter behind a **feature flag** as a kill switch, with metrics on how often it fires and for whom.
- **Dark launch** first: log what it *would* block, tune thresholds so legitimate usage passes, and sometimes help heavy customers change their code before enforcing.

## Where to put the limiter

There are three main placements, and the choice decides what the limiter can see and how accurate it is.

~~~dg-compare Three places to run a rate limiter: what it can see, what it costs, and how accurate the shared count is.
[In-process]
request -> app server -(local counter)-> handler
+ no extra hop, microseconds
+ keeps working if the network fails
- each of 20 servers counts alone
- user gets about 20x the limit
[Dedicated service]
request -> app -(RPC)-> limiter service -> store
+ global view, rich business context
- extra hop on every request
- another service that can fail
[*Gateway + shared store]
request -> gateway -(Lua script)-> Redis -> app
+ rejects before the app does any work
+ one place for every service's rules
- sees only HTTP data: IP, headers, token
~~~

| Placement | Good | Bad |
| --- | --- | --- |
| In each app server (in-process) | fastest, no extra hop | each server sees only its share; with 20 servers a user effectively gets about 20× the limit |
| Dedicated rate-limit service | global view, can use business context | extra network hop per request, another thing that can fail |
| API gateway or load balancer | rejects bad traffic before it reaches the app; most common | only sees HTTP data (IP, headers, token), not business details |

The usual choice is the **gateway**, with shared state in **Redis** (or a similar fast store) so all gateway instances count the same user together. In-process limiting still has a place: as a cheap, coarse **local** guard (for example, each server allows at most 2× its fair share) that keeps working if Redis is unreachable. Many systems use both: a local limit for protection, a global limit for fairness.

Why not just divide the limit by the number of servers? Because load balancers do not spread one user's traffic evenly. With sticky sessions or few connections, all of one user's requests may land on one server, which would reject them at a twentieth of the limit while the other 19 sit idle.

## Which key to limit by

The **key** is what the counter is attached to. It decides who shares a limit.

~~~dg-matrix Choosing the rate-limit key: each is right for a different kind of traffic and wrong for another.
Key | Works before login? | Fair per person? | Survives IP changes? | Matches billing? | Easy for attacker to rotate?
User ID | no | yes | yes | partial | no, needs accounts
API key | no | partial, one key many servers | yes | yes | no, keys are issued
IP address | yes | no, NAT groups many users | no | no | yes, botnets and proxies
Target account | yes | yes | yes | no | no, targets are fixed
~~~

| Key | Use when | Watch out for |
| --- | --- | --- |
| **User ID** (from the session or JWT) | logged-in users; fairest per person | does not help on unauthenticated endpoints |
| **API key** | developer APIs, per customer or per app; matches billing and plans | one customer may run many servers on one key; leaked keys |
| **IP address** | anonymous traffic: login, signup, public pages; first line against floods | NAT: an office, university or mobile carrier puts thousands of users behind one IP, so set IP limits higher; attackers rotate IPs |

**NAT** (network address translation) is the big trap with IP limits. A mobile carrier's carrier-grade NAT may put tens of thousands of phones behind a few hundred public IPv4 addresses. A limit of 10 signups per hour per IP would block a whole city. With IPv6, the opposite problem appears: one user can have billions of addresses (a /64 block), so limit by the /64 prefix, not the full address.

Real systems **layer** several keys, and the most restrictive rule that applies wins.

~~~dg-tree A layered rule set: a request is checked against every rule that applies, and the strictest decides.
Incoming request
  Global | 50,000 req/s across all clients
  Unauthenticated routes
    Per IP | 300 req/min
    Login: per target account | 5 failures/min
    Signup: per IP /24 or /64 | 20 per hour
  Authenticated routes
    *Per user ID | 100 req/s, burst 200
    Per API key, by plan | free 60/min, paid 6,000/min
    Expensive endpoint: per user | 2 exports per hour
~~~

Login is the classic example of needing two keys. Limiting by IP alone misses a botnet that tries one password per IP against the same account from 10,000 IPs. Limiting by target account catches that, while the IP limit catches one IP trying many accounts (credential stuffing).

## Making it correct and fast

### Atomic updates

Reading the counter, computing the new value and writing it back must happen as one step. Otherwise two concurrent requests both read "99 of 100", both pass and both write 100: one extra request slips through, and under heavy concurrency many do. In Redis, do the whole check in a **Lua script**, which runs atomically on the server, or use a single atomic command (`INCR` for a fixed window counter).

~~~dg-seq Two gateways check the same user against one Redis key; the Lua script makes each check atomic.
Gateway 1 > Redis : EVAL limiter.lua key=rl:user:42 (rate 100/s)
Redis > Redis : refill tokens, take 1, 0 left
Redis -> Gateway 1 : allowed, remaining 0
Gateway 2 > Redis : EVAL limiter.lua key=rl:user:42
Redis > Redis : refill (2 ms passed, 0.2 tokens), none to take
Redis -> Gateway 2 : rejected, retry after 8 ms
Gateway 2 > Gateway 2 : 429, Retry-After: 1
~~~

### Scale and latency

One Redis instance handles roughly 50,000 to 100,000 limiter checks per second (each check is several operations inside the script). The check adds a network round trip, typically 0.2 to 1 ms in the same datacentre. For 1 million requests per second, shard by client key across about 10 to 20 shards; Redis Cluster does this with its 16,384 hash slots, and because all of one key lives on one shard, the Lua script still sees the whole count.

<div class="ex">
<div class="lab">Worked example</div>

**sizing the limiter store**

The API serves 400,000 requests per second at peak from 8 million active users. Each request needs one limiter check against the per-user key and, for 20% of requests on unauthenticated routes, one more against an IP key: 480,000 checks per second.

At about 60,000 checks per second per Redis shard with headroom, you need 480,000 ÷ 60,000 = 8 shards; use 12 so losing one shard (and its replica failing over) does not overload the rest.

Memory: a token bucket stores two numbers per key, about 100 bytes with Redis overhead. 8 million users plus 3 million IP keys is 11 million keys × 100 bytes ≈ 1.1 GB. Set a TTL on each key (say twice the refill time) so idle users disappear on their own.

</div>

### Hot keys

One huge customer sending 50,000 requests per second on one API key puts all that load on one Redis shard. Options: give that key a local pre-limiter on each gateway that only forwards a sample to Redis, split its limit into N sub-keys (`key:0` to `key:7`, each with an eighth of the limit, chosen at random per request), or batch: each gateway takes tokens in blocks of 100 and spends them locally.

### Fail open or closed?

Stripe fails open: a limiter outage should not become an API outage. A social API might choose to fail closed, because Redis failures often happen during traffic spikes, exactly when backends need protection. Payments and security endpoints (login, password reset) usually fail closed or fall back to a strict local limit. Either way, alert when the fallback activates.

### Tell the client

Return 429 with headers such as `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset` and `Retry-After`. Reject immediately rather than queueing, so the client's own timeout and backoff logic works. Well-behaved clients then retry with exponential backoff and jitter (lesson 02.09).

### Configuration

Rules change often (a customer upgrades, an attack starts). Store them in a rules table or config service; gateways poll it about every 30 seconds and cache rules locally, with pushes for urgent changes such as blocking an attacking IP range.

<div class="ex">
<div class="lab">Worked example</div>

**the 20-server mistake**

A team limits each user to 100 requests per minute with an in-memory counter on each of 20 app servers behind a round-robin load balancer. A scraper sends 1,800 requests per minute. Each server sees about 90, under its 100, so nothing is blocked: the effective limit is 2,000.

Moving the counter to Redis at the gateway, keyed by user ID, blocks the scraper after 100. A legitimate user who sends bursts of 30 is unaffected.

</div>

## Common mistakes

- **Per-server counters** behind a load balancer, giving each client N times the limit.
- **Read-then-write without atomicity**, letting concurrent requests slip through.
- **IP-only limits** that block whole offices and mobile networks while botnets walk past.
- **Trusting `X-Forwarded-For` blindly**: clients can set it. Use the address your own edge proxy recorded.
- **No kill switch or dark launch**, so a mistuned rule blocks your biggest customer on day one.
- **Limiting only request count** when cost varies wildly per request; weight expensive calls or add a concurrency limit.

## In an interview

Say where the limiter lives (gateway, shared Redis, Lua for atomicity), what the keys are (user ID, API key, IP with NAT-aware limits, layered), what happens when Redis fails (fail open with a local fallback) and what the client sees (429 with Retry-After). Then mention that load shedding is a separate, system-wide mechanism.

<div class="take">
<div class="lab">Key takeaway</div>

rate limit per client and load shed per system. Put the limiter at the gateway with shared counters in Redis, updated atomically and sharded by key. Limit by user ID when logged in, API key for developer APIs, and IP for anonymous traffic (with NAT-friendly limits), layering several keys with the strictest winning.

</div>
