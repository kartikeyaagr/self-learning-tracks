---
id: "02.06"
course: "system-design-101"
module: "02"
slug: "load-balancers"
title: "Load balancers"
keyIdeas: ["Load balancers spread requests, detect unhealthy backends via health checks, and give clients one stable address.", "Algorithms: round robin (simple, equal servers), weighted round robin (different sizes), least connections (varying request durations), IP hash or consistent hash (stickiness), random of two choices (good spread, cheap).", "L4 balancers are fast and protocol-agnostic. L7 balancers can terminate TLS, route by URL and retry failed requests.", "The balancer itself must be redundant: active-passive pairs with a floating IP, or DNS/anycast across several.", "Least connections beats round robin when request times vary widely, because round robin keeps sending work to a server stuck on slow requests."]
question: "When does least-connections beat round robin?"
answer: "Least connections beats round robin when requests vary widely in cost or duration, or when connections are long-lived, like WebSockets or streaming. Round robin keeps handing new work to a server that is stuck on slow requests, so requests queue behind them while other servers sit idle and p99 latency soars; least connections sees that server's high in-flight count and routes around it, which keeps queues short, balances live connections, and keeps tail latency low."
sources: [{"title": "Sam Rose: Load Balancing (interactive)", "url": "https://samwho.dev/load-balancing/", "note": "Animated comparison of round robin, least connections and more."}, {"title": "System Design Primer: Load balancer", "url": "https://github.com/donnemartin/system-design-primer#load-balancer", "note": "L4 vs L7, active-passive vs active-active."}, {"title": "NGINX docs: HTTP load balancing", "url": "https://docs.nginx.com/nginx/admin-guide/load-balancer/http-load-balancer/", "note": "Real config for upstreams and methods."}]
tags: ["Code"]
tools: ["nginx"]
---

## Why load balancers exist

A **load balancer** sits in front of a group of servers (the **backends**, or the **pool**) and decides which one handles each request or connection. It gives you:

- **No traffic to unhealthy servers**: it detects failed backends and routes around them.
- **No single overloaded server**: work is spread so queues stay short.
- **No single server as a point of failure**: losing one backend costs a fraction of capacity, not the service.
- **TLS termination**: the balancer decrypts traffic, so backends skip the encryption work and certificates live in one place.
- **Painless deploys and scaling**: drain a backend (stop new requests, let current ones finish), update it, add it back; add servers under load and remove them at night.
- **Optional affinity**: pin a client or a key to one server when that helps (sessions, caches).

The catch is that a single load balancer is itself a single point of failure. Load balancers therefore run in pairs or clusters:

- **Active-passive**: one serves traffic while a standby watches its heartbeats. If heartbeats stop, the standby takes over the shared virtual IP address (for example with the VRRP protocol, as keepalived does), usually within a few seconds.
- **Active-active**: several balancers all serve traffic. DNS returns several addresses, or routers spread packets across them using equal-cost multipath routing (ECMP). Managed cloud load balancers work this way behind a single name.

Load balancing only works well when backends are **stateless**. If session data lives on one server, the balancer must pin users to it, and losing that server loses the sessions. Keep that state in a shared store such as Redis instead (lesson 01.08).

## Layer 4 vs Layer 7

- **Layer 4** load balancers look at IP addresses and ports only. They pick a backend when a connection opens and forward its packets (often by rewriting addresses with NAT, or by passing packets through untouched with direct server return). They are cheap and fast, handle millions of packets per second, and work for any TCP or UDP protocol: databases, MQTT, custom protocols. But every request on a connection goes to the same backend, and they can't route on content.
- **Layer 7** load balancers terminate the client's connection, read the HTTP request, pick a backend **per request**, and open (or reuse) their own connection to it. They can route by path, host, header or cookie (send `/video` to video servers, `/billing` to hardened servers), retry a failed idempotent request on another backend, add headers, and compress responses. This costs more CPU and adds a little latency, typically well under a millisecond.

~~~dg-compare Layer 4 balances connections using only addresses and ports; Layer 7 reads each HTTP request and can route on its content.
[Layer 4]
Packet -(IP and port only)-> L4 LB -(whole connection)-> Server 2
+ Very fast, any protocol
+ Can pass TLS through untouched
- One backend per connection
- Can't route on path or headers
[Layer 7]
HTTP request -(path, headers, cookie)-> L7 LB -(/api/*)-> API pool
+ Per-request balancing and routing
+ Retries, header rewriting, TLS termination
- More CPU per request
- HTTP-family protocols only
~~~

The connection-versus-request difference matters a lot for long-lived connections. An HTTP/2 or gRPC client keeps one connection open for hours and sends thousands of requests over it (lesson 02.03). Behind a Layer 4 balancer, all of those requests land on one backend. With a few large clients, a few backends get almost all the load. A Layer 7 balancer, or client-side load balancing where the client itself spreads calls across a list of backends, fixes this.

## The algorithms

The algorithm decides which backend gets the next request. The differences show up clearly when request costs vary.

**Round robin** sends to each server in turn: A, B, C, A, B, C. It is fair when every request costs the same and every server is the same size. When some requests are expensive, a server that happens to receive several slow ones in a row builds a queue while others sit idle, because round robin doesn't look at how busy anyone is.

**Weighted round robin** gives bigger servers a bigger share. With weights 5 and 1, the first server gets 5 of every 6 requests. Useful when the pool mixes instance sizes, or to send a small share of traffic to a canary.

**Dynamic weighted round robin** adjusts weights from observed latency or errors, so a server that slows down automatically gets less.

**Least connections** sends each request to the server with the fewest requests currently in flight. A server stuck on slow work naturally has more open requests, so it gets skipped until it catches up. In simulations with varying request costs, least connections handles overload better than round robin, dropping the fewest requests.

**Least time / PEWMA** (peak exponentially weighted moving average) scores each server by its recent latency multiplied by its in-flight requests, reacting quickly to latency spikes. It tends to give the best latency, at the cost of more complexity and tuning.

**Hashing** (on client IP, or a key such as user ID or URL) sends the same key to the same server every time. That's useful for caches and sessions. Plain `hash % N` remaps almost every key when N changes, so balancers offer **consistent hashing** (NGINX's `hash $key consistent`), where adding a server only moves a small share of keys (lesson 02.07).

**Power of two random choices** picks two servers at random and sends to the less loaded of the two. It sounds too simple to work, but it is remarkably effective: the maximum queue length drops dramatically compared with picking one at random. It is especially useful when many load balancers share the same backends, because no balancer can see the others' counts, and "least connections" computed on partial information makes all of them pile onto the same apparently idle server.

~~~dg-timeline Round robin with uneven request costs. Server A receives two slow reports in a row and builds a queue, while B and C finish quickly and sit idle.
[Server A] 0 ms | report request, 900 ms
[Server B] 0 ms | page request, 20 ms
[Server C] 0 ms | page request, 20 ms
[Server A] 30 ms | *2nd report queued behind the 1st
[Server B] 30 ms | idle
[Server C] 30 ms | idle
[Server A] 60 ms | 3rd request queued, waits ~1.8 s
~~~

An important detail: in such simulations, round robin often has the best **median** latency but poor **p95 and p99**, because most requests land on idle servers while a few wait behind slow ones. Averages hide the problem; always judge an algorithm on its tail.

<div class="ex">
<div class="lab">Worked example</div>

**round robin versus least connections with mixed requests**

Three servers each handle one request at a time. 90% of requests take 10 ms; 10% are reports that take 1,000 ms. Requests arrive every 50 ms, so on average each server is about 70% busy (an average request costs 0.9 × 10 + 0.1 × 1,000 = 109 ms).

Round robin: every third request goes to server A regardless. When A gets a report, the next requests assigned to A queue behind it: A receives one every 150 ms, so about 6 or 7 requests pile up during the report and wait up to a second each. With reports arriving every half second or so across the pool, this keeps happening, so p99 is around a second or more, even though B and C are often idle.

Least connections: while A is busy with the report, its in-flight count is 1 and B and C are usually at 0, so new requests go to them. Only when all three are busy does anyone queue. The p99 drops to roughly the cost of a report plus a little, and short requests rarely wait behind long ones.

</div>

## Health checks and recovery

A load balancer is only as good as its knowledge of which backends are healthy:

- **Passive checks**: watch real traffic. NGINX counts failures and marks a server down after `max_fails` failures within `fail_timeout`, then tries it again after that period.
- **Active checks**: probe an endpoint such as `/health` on a schedule (every 5 to 10 seconds), with thresholds: unhealthy after 3 consecutive failures, healthy again after 2 successes. The thresholds stop one dropped probe from flapping a server in and out.
- **What to check**: a liveness endpoint that only says "the process is running" misses a server whose database connection is broken. A deep check that tests every dependency can take the whole fleet out when a shared dependency blips. A common compromise is to check the server's own critical resources and leave shared dependencies to circuit breakers.
- **Slow start**: a server that comes back (or a new one) ramps up its weight gradually, for example over 30 seconds, so it isn't overwhelmed while its caches and connection pools are cold, and knocked over again.
- **Connection draining**: when removing a server, stop new requests but let in-flight ones finish (for example within 30 seconds).
- **Backup servers** only receive traffic when all primaries are down.

~~~dg-state A backend's life as the load balancer sees it. Thresholds and slow start stop servers from flapping in and out.
[*] -> Healthy
Healthy -> Suspect : 1 failed check
Suspect -> Healthy : next check passes
Suspect -> *Unhealthy : 3 failures in a row
Unhealthy -> Slow start : 2 checks pass
Slow start -> Healthy : weight ramps to full over 30 s
Healthy -> Draining : removed for deploy
Draining -> Removed : in-flight requests done
~~~

**Sticky sessions** use a cookie (or the client IP) to pin a client to one server. They're convenient for legacy apps that keep sessions in memory, but they make load uneven (one heavy user stays on one server) and hurt when that server dies or is drained.

| Algorithm | Best when |
| --- | --- |
| Round robin | Requests and servers are uniform |
| Weighted round robin | Servers differ in size, or for canary traffic |
| Least connections | Request cost varies a lot, or connections are long-lived (WebSockets) |
| Least time / PEWMA | You care most about tail latency and can afford complexity |
| Consistent hash | Server holds per-key state, such as a cache |
| Power of two choices | Many balancers share the same backends |

~~~dg-bars Open WebSocket connections per server after six hours. Round robin balanced connection openings, not live connections; least connections balances what is actually open. (Illustrative numbers.)
Server 1, round robin | 14200 | 14,200
Server 2, round robin | 6100 | 6,100
Server 3, round robin | 9700 | 9,700
*Servers 1 to 3, least connections | 10000 | ~10,000 each
~~~

<div class="ex">
<div class="lab">Worked example</div>

**chat servers with WebSockets**

A chat service runs 3 servers and holds about 30,000 open WebSockets in total. Each socket stays open from minutes to many hours.

Round robin counts connections evenly at the moment they open, but some users disconnect after a minute while others stay all day, and a server restarted for a deploy comes back empty while the others keep their sockets. After a few hours one server might hold 14,000 live sockets and another 6,000. The busy one runs hot on memory and CPU.

Least connections sends each new socket to the server with the fewest open ones, so the counts converge back towards 10,000 each, and a freshly restarted server fills up quickly instead of staying underused.

</div>

## Common mistakes

- **One load balancer with no redundancy.**
- **Layer 4 in front of gRPC or HTTP/2**, pinning heavy clients to one backend.
- **Judging algorithms by average latency.** Look at p99.
- **Health checks that are too shallow or too deep**, missing broken servers or ejecting the whole fleet together.
- **No slow start**, so recovered servers get knocked straight back down.
- **Sticky sessions instead of a shared session store.**

## In an interview

Draw a load balancer in front of every stateless tier, say it is redundant, and name the algorithm with a reason: "least connections, because report requests vary widely in cost" or "consistent hashing on user ID, because each server caches user data". Mention health checks and draining for deploys.

<div class="take">
<div class="lab">Key takeaway</div>

round robin is fine for uniform work; least connections wins when request cost varies or connections are long-lived, and hashing when servers hold per-key state. Balance requests, not just connections, for HTTP/2 and gRPC, make the balancer itself redundant, and judge algorithms on p99, not the average.

</div>
