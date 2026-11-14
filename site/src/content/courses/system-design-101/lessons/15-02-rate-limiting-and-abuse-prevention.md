---
id: "15.02"
course: "system-design-101"
module: "15"
slug: "rate-limiting-and-abuse-prevention"
title: "Rate limiting and abuse prevention"
keyIdeas: ["Design a rate limiter shared by many API servers.", "Store counters in a central fast store (Redis), keyed by user and window. Each server does an atomic increment-and-check (Lua script) per request.", "Algorithms: token bucket or sliding window counter.", "Latency: one Redis round trip per request; reduce with local pre-checks or batching for very high rates.", "Failure: if Redis is down, decide to fail open (allow) or closed (deny). Most choose open plus a local fallback limit.", "Return 429 with headers. Rules (limits per endpoint and plan) come from config."]
question: "How do 20 API servers share one rate limit per user?"
answer: "They share state. No server keeps its own counter; instead every request does an atomic check-and-decrement against the same per-user key in a central store like Redis: a token bucket (tokens and last refill time) updated by a Lua script that Redis runs as one step, so two servers cannot both spend the last token. Redis is sharded by the user key, so each user's bucket lives on exactly one shard, with a replica for failover, and a policy decides whether to fail open (with a coarse local fallback) if Redis is unreachable. However the load balancer spreads a user's requests, all 20 servers draw from the same bucket."
sources: [{"title": "Hello Interview: Distributed rate limiter", "url": "https://www.hellointerview.com/learn/system-design/problem-breakdowns/distributed-rate-limiter", "note": "Full design."}, {"title": "Stripe: Scaling your API with rate limiters", "url": "https://stripe.com/blog/rate-limiters", "note": "Production view."}]
tags: ["Design"]
tools: ["PostgreSQL"]
---

## The problem: twenty servers, one limit

Say the rule is 100 requests per minute per user, and your API runs on 20 servers behind a load balancer. If each server counts on its own, a user whose requests are spread evenly gets 100 on *each* server: up to 2,000 a minute. Each server sees only about one twentieth of a user's traffic, so a local counter cannot enforce a global limit. Dividing the limit by 20 does not work either, because load balancers do not spread one user's traffic evenly: a user with one keep-alive connection may send everything to one server and be blocked at 5 a minute.

The fix is **shared state**: every server consults the same counter for the same user. This lesson designs that system end to end, using the placement and key choices from lesson 14.11 and the algorithms from lesson 14.12.

## Requirements

The scenario: a social media API at about **1 million requests per second** from 100 million daily users.

- **Functional**: identify clients by user ID, IP or API key; support configurable rules per endpoint and per plan ("5 logins per minute per account", "free plan: 60 requests per minute"); reject over-limit requests with a clear response.
- **Non-functional**: the check adds under **10 ms** (ideally about 1 ms); the limiter itself must be highly available, because if every request depends on it, its outage is your outage; it can be **eventually consistent**: being off by a few requests is acceptable, being off by 20× is not.

## Design: gateway plus Redis

~~~dg-flow Every gateway checks the same per-user bucket in Redis, so 20 servers enforce one limit; rules are cached locally and refreshed in the background.
[Request path] Client -(user 42)-> Load balancer -> Gateway 7 -(Lua on rl:user:42)-> *Redis shard 3 -(allow)-> App servers
[Reject path] Redis shard 3 -(no tokens)-> Gateway 7 -(429, Retry-After)-> Client
[Control plane] Rules DB -(poll every 30 s)-> Gateway rule cache -(metrics)-> Dashboards and alerts
~~~

- **Place it at the API gateway** so rejected traffic never reaches the app servers. A rejected request should cost you about a millisecond of gateway CPU and one Redis call, nothing more.
- **Keep the counters in Redis**, keyed by rule and client (`rl:api:user:42`). All 20 gateway instances read and update the same key, so together they enforce one limit.
- **Algorithm: token bucket**, which needs only two fields per client: `tokens` and `last_refill`. On each request, add the tokens earned since `last_refill` (capped at capacity), then take one if available. Set an `EXPIRE` of about an hour so idle clients' keys disappear by themselves.

### How a token bucket behaves over time

A bucket with capacity 10 and refill rate 1 per second (that is, 60 per minute with bursts of 10) gives this behaviour:

~~~dg-timeline One user's token bucket (capacity 10, refill 1 per second): a burst is absorbed, then the steady rate takes over.
t = 0 s | Bucket full: 10 tokens
t = 0.1 s | Burst of 10 requests, all allowed, 0 tokens left
t = 0.2 s | *11th request rejected: 429, Retry-After 1
t = 1.1 s | 1 token refilled, 1 request allowed
t = 5 s | Steady 1 per second allowed, bucket stays near 0
t = 30 s | User idle 20 s: bucket refilled to 10 (capped)
~~~

## The race condition

The naive version reads with `HMGET`, computes in the gateway, then writes back with `HSET`. Two gateways can both read "1 token left" at the same moment, both allow, and both write 0: two requests passed when only one should have. A plain counter has the same bug: two requests read 3, both write 4, when the right answer is 5. Under a burst of 50 concurrent requests from one client, dozens can slip through.

~~~dg-seq Without atomicity, two gateways both spend the last token; a Lua script makes read, refill and take one step.
Gateway A > Redis : HMGET rl:user:42 (tokens = 1)
Gateway B > Redis : HMGET rl:user:42 (tokens = 1)
Gateway A > Redis : HSET tokens = 0 (allow)
Gateway B > Redis : HSET tokens = 0 (allow, wrong)
Gateway A > Redis : fix: EVALSHA bucket.lua rl:user:42
Redis > Redis : read, refill, take, write: nothing runs in between
Redis -> Gateway A : allowed, 0 left
~~~

Fixes:

- **A lock** around the read-modify-write: correct but slow, with an extra round trip and lock contention on hot keys.
- **A Lua script** that Redis runs atomically: read, refill, decide and write happen as one step on the Redis server, and Redis (being single-threaded per shard) runs nothing else in between. This is the usual choice: one round trip, no lock.
- **Single atomic commands** where the algorithm allows: `INCR` for a fixed window, which is atomic on its own.

The script is short:

```lua
-- KEYS[1] = bucket key; ARGV = capacity, rate per ms, now in ms
local cap, rate, now = tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3])
local b = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(b[1]) or cap
local ts = tonumber(b[2]) or now
tokens = math.min(cap, tokens + (now - ts) * rate)
local allowed = tokens >= 1
if allowed then tokens = tokens - 1 end
redis.call('HSET', KEYS[1], 'tokens', tokens, 'ts', now)
redis.call('PEXPIRE', KEYS[1], 3600000)
return { allowed and 1 or 0, tokens }
```

Gateways load it once with `SCRIPT LOAD` and call it by its hash with `EVALSHA`, so the script text is not sent on every request. One subtlety: `now` comes from the gateway's clock, so gateways with skewed clocks can refill a bucket early or late. Either keep clocks in sync (NTP keeps them within milliseconds) or read the time inside the script with Redis's `TIME` command.

**Sticky sessions** (routing each user always to the same gateway, which then counts locally) are an alternative without shared state. But they make load uneven, and when a gateway fails its users move elsewhere with fresh counters. They do not scale or handle failures well.

## Scaling and failure

### Throughput

One Redis instance handles roughly 100,000 to 200,000 simple operations per second; a Lua check does several operations, so budget 50,000 to 100,000 checks per second per shard. For 1 million per second, **shard by client key** over about 10 to 20 shards. Redis Cluster does this with 16,384 hash slots: the key is hashed to a slot and each shard owns a range of slots. All of one client's data lives on one shard, so the Lua script sees the whole bucket.

<div class="ex">
<div class="lab">Worked example</div>

**sizing the cluster**

Peak load is 1.2 million requests per second. 70% are authenticated (one check against the user key); 30% are anonymous and need an IP check, and logins (1% of traffic) also check a per-account key: about 1.2M × (1 + 0.01) ≈ 1.21 million checks per second.

At a planned 70,000 checks per second per shard, that is 18 shards; round up to 24 so that one shard failing over (and briefly slow) does not push the others past their limit. Each shard gets one replica: 48 Redis processes.

Memory: 100 million daily users plus 30 million IP keys, about 100 bytes each with overhead, is 13 GB in total, about 550 MB per shard. Memory is not the constraint; operations per second are.

Latency: inside one datacentre a Redis round trip is 0.2 to 0.5 ms, well inside the 10 ms budget.

</div>

### Availability

Each shard gets a replica with automatic failover. Replication is asynchronous, so a failover may lose the last few hundred milliseconds of counter updates. For a rate limiter that means a few clients get a slightly bigger allowance for a moment, which the requirements accept.

### Redis down: fail open or closed?

If the limiter cannot reach Redis, it must decide what to do with the request. There is no universally right answer; decide per rule.

~~~dg-matrix Fail open or fail closed when the limiter's store is unreachable: decide per endpoint, and alert either way.
Endpoint type | Fail open | Fail closed | Local fallback limit
Public read API | yes, outage of limiter is not an API outage | no | yes, coarse per-gateway cap
Login and password reset | no, opens brute force window | yes | yes, strict
Expensive exports | partial | partial | yes, per-gateway concurrency cap
Payments and SMS sending | no, costs real money | yes | yes
~~~

Stripe fails open, letting traffic through, so a limiter bug cannot take the API down. A social API might fail closed on some routes, because Redis failures often coincide with traffic spikes, exactly when backends most need protection. A good compromise is a **local fallback**: each gateway keeps a coarse in-memory limit (say, its fair share times two) that applies only while Redis is unreachable. Either way, alert when the fallback activates.

### Latency and regions

Use pooled, persistent connections to Redis, and pipeline when a request needs several checks. Deploy Redis in each region and count per region: a user served from Europe and Asia at once gets up to two allowances, which is acceptable for rate limits. Synchronising counters across regions on every request would add 100 ms or more, far over budget.

### Hot keys

An abusive client hitting one key can overload a shard (all its checks land on the same Redis process). Put repeat offenders on a **blocklist** cached in each gateway's memory, so their requests are rejected without touching Redis, and handle volumetric floods at the edge (Cloudflare, AWS Shield) before they reach your gateways.

### Rules

Rules live in a config table that gateways poll about every 30 seconds and cache locally, like Lyft's rule files (for example, domain `auth`, key `login`: 5 per minute). Urgent changes, such as blocking an attacking IP range, are pushed. Each rule needs an owner, a reason and a dark-launch mode that logs instead of blocking.

## Responses

Reject immediately with **429 Too Many Requests** and headers so clients can back off:

- `X-RateLimit-Limit`: the limit for this rule;
- `X-RateLimit-Remaining`: requests left in the current period;
- `X-RateLimit-Reset`: when the allowance resets;
- `Retry-After`: seconds to wait (add a little random jitter so blocked clients do not all return at once).

Do not queue rejected requests: queueing uses memory, makes latency unpredictable and encourages clients to pile on more retries. Well-behaved clients use exponential backoff with jitter (lesson 02.09).

<div class="ex">
<div class="lab">Worked example</div>

**a request traced end to end**

User 42 sends `GET /feed`. Gateway 7 extracts the user ID from the JWT, finds two applicable rules in its local cache (per user 100 per minute with burst 20; global per endpoint 200,000 per second), and pipelines two `EVALSHA` calls to the shards owning `rl:feed:user:42` and `rl:feed:global`.

Both return allowed in 0.4 ms. The user's bucket was full, so the gateway adds `X-RateLimit-Remaining: 19` and forwards the request.

Thirty seconds later a script on the same account sends 300 requests in a second. The bucket has refilled to its cap of 20 by then, so the first 20 pass (the burst), the rest get 429 with `Retry-After: 1`. The app servers never see them, and a counter of rejections per user feeds an alert if one client is rejected more than 10,000 times an hour.

</div>

## Beyond per-user limits

Rate limiting is one part of protecting an API. Stripe runs, besides the per-user **request rate limiter**, a **concurrent requests limiter** (at most about 20 in-flight requests per user, for slow endpoints) and two **load shedders** that act on whole-system health: one keeps about 20% of capacity for critical calls like creating charges, the other sheds test-mode traffic first when workers run out. Rate limiting stops one client hogging resources; load shedding keeps the most important traffic alive when the whole system is overloaded. And for abuse that stays under every limit (slow credential stuffing from 50,000 IPs), you need detection on patterns, covered in lesson 14.10.

## Common mistakes

- **Local counters** behind a load balancer: 20 servers, 20 times the limit.
- **Read, compute, write** in the gateway without atomicity.
- **One Redis instance** for a million checks per second, with no replica.
- **Global cross-region counters** on the request path, blowing the latency budget.
- **No decision on fail open versus fail closed** until the first Redis outage makes it for you.

## In an interview

Start from the 20-server problem, then present: gateway placement, Redis token buckets keyed by client, a Lua script for atomicity, sharding by key with replicas, per-region counting, a fail-open policy with a local fallback, 429 with Retry-After, and rules in a config store. Put numbers on throughput and memory. If time allows, separate rate limiting from load shedding.

<div class="take">
<div class="lab">Key takeaway</div>

local counters on 20 servers enforce 20× the limit. Put the limiter at the gateway, keep one token bucket per client in Redis, update it atomically with a Lua script, shard Redis by client key with replicas, count per region, choose fail-open or fail-closed deliberately per endpoint, and answer with 429 and Retry-After.

</div>
