---
id: "06.10"
course: "system-design-101"
module: "06"
slug: "circuit-breakers-and-timeouts"
title: "Circuit breakers and timeouts"
keyIdeas: ["A circuit breaker wraps calls to a dependency and tracks failures. **Closed**: calls pass. **Open**: after too many failures, calls fail immediately. **Half-open**: after a cooldown, a few test calls decide whether to close again.", "Failing fast frees threads and connections instead of waiting on a dead service, and gives the dependency time to recover.", "Combine with timeouts (every call), retries with backoff (few), bulkheads (separate pools per dependency) and fallbacks.", "Cascading failure: one slow service makes callers queue up, which makes them slow, and the slowness spreads. Breakers and timeouts stop the spread."]
question: "Why does a circuit breaker make the system fail faster, and why is that good?"
answer: "Once failures (errors, timeouts or slow calls) cross a threshold, the breaker opens and rejects calls immediately, in microseconds, instead of letting every call wait for a timeout. That is good because the request would fail anyway: failing fast frees the caller's threads and connections so it keeps serving everything that does not need the broken dependency, stopping a cascading failure; it gives the struggling dependency room to recover instead of being hammered by requests and retries; it gives users and upstream services an immediate error they can handle with a fallback such as cached or degraded data; and half-open trial calls close the breaker automatically once the dependency is healthy."
sources: [{"title": "Martin Fowler: Circuit Breaker", "url": "https://martinfowler.com/bliki/CircuitBreaker.html", "note": "Closed, open, half-open."}, {"title": "Google SRE book: Addressing Cascading Failures", "url": "https://sre.google/sre-book/addressing-cascading-failures/", "note": "How one slow service takes down others."}]
---

## The problem: slow is worse than dead

When a service you depend on **crashes**, calls to it fail fast: the connection is refused in a millisecond, you get an error, and you move on. When it becomes **slow** or hangs, something much worse happens. Every call waits until a timeout. While it waits, the calling thread (or connection, or memory for the request) is tied up doing nothing.

Callers have finite resources. A service with a pool of 200 worker threads, each normally busy for 50 ms per request, can serve 200 / 0.05 = 4,000 requests per second. If one dependency starts taking 10 seconds, every request that touches it holds a thread for 10 seconds, and the pool empties within moments. Now requests that **never needed** the slow dependency also queue, because there are no free threads. The caller is effectively down. Its own callers then see it as slow, and the same thing happens to them.

This chain reaction is a **cascading failure**: one slow component takes down everything upstream of it. It is one of the most common shapes of large outages.

~~~dg-timeline How one slow database cascades into a full outage when nothing limits waiting.
0 s | Database latency rises from 5 ms to 5 s
2 s | Service B's 100 threads all wait on the database
4 s | Service A's calls to B now hang too
6 s | *A's thread pool is full; requests that never touch B also stall
10 s | Load balancer health checks on A time out
15 s | *A marked unhealthy everywhere; whole site down
~~~

## Timeouts first

Every remote call needs a **timeout** (lesson 02.09). Without one, a hung dependency holds resources forever, and no amount of other cleverness helps. Setting timeouts well is harder than it looks:

- **Base them on measurements.** Set the timeout a little above the dependency's healthy p99 or p99.9 latency. If p99 is 80 ms, a 200 ms timeout is reasonable; a 30 second default is not.
- **Avoid deadlines far above normal latency.** Suppose normal latency is 50 ms and the timeout is 100 s. If 5% of requests start hanging, each one holds a thread for 100 s while healthy requests take 50 ms. Hung requests quickly come to occupy nearly every thread, and most of **all** requests (healthy ones included) end up failing. Latency becomes **bimodal**: either fast, or stuck until the huge deadline.
- **Propagate deadlines.** Pass the remaining time budget down the call chain (gRPC does this with deadlines in request metadata). If the user's request has 300 ms left and service A has used 250 ms, it tells service B "you have 50 ms". B does not start expensive work that can no longer be useful.
- **Cancel** work that is no longer needed. When the caller gives up, the callee should stop too, rather than finishing a query whose answer nobody will read.

<div class="ex">
<div class="lab">Worked example</div>

**why a 100 s timeout is worse than a 1 s one**

A service has 200 threads and receives 2,000 requests per second. Normal latency is 50 ms, so on average 2,000 × 0.05 = 100 threads are busy.

A dependency starts hanging 5% of requests. With a 100 s timeout, 100 hanging requests arrive per second and each holds a thread for 100 s. Within about 2 seconds, all 200 threads are stuck. From then on, almost every request, healthy or not, waits for a thread and times out upstream: close to 100% failure from a 5% problem.

With a 1 s timeout, the 100 hanging requests per second each hold a thread for 1 s, occupying about 100 threads. Healthy requests need another 95 threads. Total 195 of 200: tight, but the service keeps working and only the 5% fail.

</div>

Timeouts bound how long each call waits. But a timeout still makes every call to a dead dependency wait for the full timeout, and still sends every request to a service that is already struggling. That is what circuit breakers fix.

## The circuit breaker

A **circuit breaker** wraps calls to one dependency and watches their outcomes. Like an electrical breaker, it **trips** when something is wrong and stops the flow entirely, then allows a careful reset. It has three states.

- **Closed** (normal): calls go through. The breaker counts failures: errors, timeouts, rejections.
- **Open** (tripped): once failures cross a threshold, calls fail **immediately** without contacting the dependency. No waiting, no network traffic.
- **Half-open** (testing): after a **reset timeout** (a cool-down, for example 30 s), the breaker lets a small number of trial calls through. If they succeed, it closes. If they fail, it opens again for another cool-down.

~~~dg-state A circuit breaker's three states. While open, calls fail in microseconds instead of waiting for a timeout.
[*] -> Closed
Closed -> Closed : under threshold
Closed -> *Open : failures cross threshold
Open -> Half-open : cool-down elapses
Half-open -> Closed : trial calls succeed
Half-open -> Open : a trial call fails
~~~

### Choosing the threshold

Thresholds come in two styles:

- **Consecutive failures**: open after 5 failures in a row. Simple, works for low-traffic calls.
- **Failure rate over a window**: open when at least 50% of calls fail, over a sliding window of the last 20 calls or the last 10 seconds, with a **minimum number of calls** (say 20) before the rate counts. Without the minimum, 1 failure out of 1 call is "100%" and the breaker opens on noise. Libraries such as Resilience4j also support a **slow-call rate**: count calls slower than a threshold as failures even if they eventually succeed, which is what catches a dependency that is slow rather than erroring.

### Practical notes

- **Log every state change and alert when a breaker opens.** An open breaker is a symptom of a real problem.
- **Let operators trip or reset breakers by hand**, for example to cut off a dependency during an incident.
- **Not every error should count.** A 404 "not found" or a 400 "bad request" is a normal answer from a healthy service, not a sign of sickness. Count timeouts, connection errors and 5xx responses.
- **One breaker per dependency** (and sometimes per endpoint or per host), so one sick dependency does not block calls to healthy ones.
- **Run calls on a bounded pool** (a bulkhead, below), so the breaker can also count "pool exhausted" as a failure.

~~~dg-seq A breaker opening and later recovering: once open, calls fail instantly; after the cool-down, one trial call decides.
Caller > Breaker : call 1 to 20 (half time out)
Breaker > Breaker : 50% failures, OPEN
Caller > Breaker : call 21
Breaker -> Caller : fail fast in 0.01 ms, use fallback
Breaker > Breaker : 30 s later, HALF-OPEN
Breaker > Dependency : trial call
Dependency -> Breaker : ok in 40 ms
Breaker > Breaker : CLOSED
~~~

## Why failing fast is good

It sounds odd to deliberately fail requests that might have succeeded. But compare the alternatives: wait 2 seconds and then fail, or fail in 10 microseconds. Either way the request fails. Failing fast is better for everyone:

- **It protects the caller.** Threads are not stuck waiting, so the caller keeps serving the many requests that do not need the broken dependency. The cascade stops here.
- **It protects the dependency.** A struggling service that keeps receiving full traffic (plus retries) cannot recover; it spends all its capacity on requests that will time out anyway. An open breaker gives it breathing room.
- **It gives clear, immediate errors.** Users and upstream services get an answer they can handle (show cached data, queue the work, show "try again later") instead of a long hang that looks like a broken page.
- **It recovers automatically.** Half-open probes detect when the dependency is healthy again, with no human involved, and they do it with a trickle of traffic rather than a flood.

### What to do when the breaker is open

A breaker returns an error; your code decides what that means. Options, roughly in order of preference:

- **Serve stale or cached data**: last known recommendations, cached prices for display.
- **Degrade the feature**: render the page without the widget.
- **Defer the work**: queue a credit card authorisation for later, accept the order, confirm by email.
- **Fail the operation** clearly, with a useful message.

~~~dg-compare A slow dependency with only timeouts versus with a circuit breaker in front of it.
[Timeouts only]
Request -(waits 2 s)-> Sick service -> Timeout error
- Every call pays the full timeout
- Threads pile up in the caller
- Sick service keeps getting hammered
[*Timeout plus breaker]
Request -(breaker open)-> Fallback in 0.01 ms -> Degraded page
+ Caller threads stay free
+ Sick service gets room to recover
+ Probes close the breaker when healthy
~~~

## The wider toolkit against cascades

Circuit breakers are one tool among several that work together:

- **Load test** to know your limits. This is the most important step: you cannot set thresholds, pool sizes or timeouts without knowing where things break.
- **Shed load** and **serve degraded results** when overloaded (lesson 02.13). Reject excess work early and cheaply rather than accepting everything and serving nobody.
- **Limit retries.** Use randomised exponential backoff, a per-request retry limit, and a **retry budget** (for example, retries may add at most 10% to traffic). Retry at **one layer only**: if three layers each make up to 4 attempts, one user request can become 4 × 4 × 4 = 64 attempts at the database, exactly when it is least able to cope.
- **Bulkheads**: separate thread pools, connection pools or capacity per dependency or per client, like the watertight compartments of a ship. If the recommendations pool fills, the payments pool is untouched.
- **Beware slow startup and cold caches.** A service that just restarted may be much slower until its caches warm, and can be knocked over again immediately.
- **Beware health checks that kill overloaded servers.** If a health check fails because a server is busy, the orchestrator restarts it, its load moves to the remaining servers, which then fail their health checks too. Health checks should test liveness, not whether the server is momentarily slow.

~~~dg-bars Worst-case attempts reaching the bottom service for one user request when every layer retries. Retrying at one layer keeps it bounded.
Retry at one layer, 3 tries | 3 | 3 attempts
Two layers, 3 tries each | 9 | 9 attempts
Three layers, 3 tries each | 27 | 27 attempts
*Three layers, 4 tries each | 64 | 64 attempts
~~~

<div class="ex">
<div class="lab">Worked example</div>

**recommendations widget**

The product page calls a recommendations service with a 150 ms timeout, through a breaker that opens at 50% errors over at least 20 calls and retries after 30 s, on its own bulkhead pool of 30 threads.

The recommendations service slows to 2 s per call. Without the breaker, every page render waits the full 150 ms timeout for the widget, and at 3,000 page views per second the 30-thread pool is constantly full, so many renders wait even longer for a thread.

With the breaker, after about 20 failed calls (well under a second at this traffic) it opens. Pages render immediately with cached bestsellers in place of personalised picks. Every 30 s a single probe checks whether recommendations are back; when one succeeds, the breaker closes and personalised widgets return. Users barely notice.

</div>

## Common mistakes

- **No timeout**, or a library default of 30 to 60 seconds.
- **Counting client errors** (404, 400) as failures, so a healthy service trips the breaker.
- **One breaker for everything**, so one bad dependency blocks calls to all of them.
- **Retrying inside an open breaker's fallback**, which defeats it.
- **Retries at every layer**, multiplying load during an incident.
- **No fallback**: the breaker opens and the page shows a raw error.

## In an interview

When your design calls another service, say: "every call has a timeout based on its p99, a circuit breaker per dependency, retries with backoff at one layer and a retry budget, and a fallback". Then explain the one line that matters most: the breaker makes failures fast so threads stay free and the sick dependency can recover.

<div class="take">
<div class="lab">Key takeaway</div>

timeouts bound how long you wait; a circuit breaker stops calling a failing dependency at all. By failing fast it keeps the caller's threads free, gives the dependency room to recover, lets callers fall back to degraded responses, and probes automatically for recovery, which stops one slow service from cascading into an outage.

</div>
