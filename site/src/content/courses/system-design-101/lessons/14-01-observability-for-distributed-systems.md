---
id: "14.01"
course: "system-design-101"
module: "14"
slug: "observability-for-distributed-systems"
title: "Observability for distributed systems"
keyIdeas: ["Observability is being able to understand what the system is doing from its outputs.", "**Metrics**: numeric time series (request rate, error rate, latency percentiles, CPU). Cheap, good for dashboards and alerts.", "**Logs**: detailed records of events. Good for investigating specific requests. Use structured logs with request IDs.", "**Traces**: follow one request across services as a tree of spans, showing where time went. Best for finding which service is slow.", "Google SRE's four golden signals: latency, traffic, errors, saturation.", "Alert on user-facing symptoms (errors, latency), not on every cause (CPU high)."]
question: "A checkout is slow. Which of logs, metrics and traces tells you which service is slow?"
answer: "Traces. Metrics show that checkout latency rose but are aggregated per service, and logs are per service and lack cross-service context. A trace follows one checkout request through every service as a tree of timed spans (connected by the propagated trace ID), so the waterfall shows directly which span, say the payment service's authorise call, takes most of the time. You then filter that service's logs by the trace ID to see why, and confirm the cause with a metric such as connection pool saturation."
sources: [{"title": "Google SRE book: Monitoring Distributed Systems", "url": "https://sre.google/sre-book/monitoring-distributed-systems/", "note": "The four golden signals."}, {"title": "OpenTelemetry: Observability primer", "url": "https://opentelemetry.io/docs/concepts/observability-primer/", "note": "Logs, metrics, traces."}]
---

## What observability means

Observability is the ability to understand what a system is doing **from the outside**, using only the data it already emits. The practical test is this: when something breaks in a way nobody predicted (an "unknown unknown"), can an engineer find the cause from existing signals, without shipping new logging first? If the first step of every investigation is "add more logging and wait for it to happen again", the system is not observable enough.

Monitoring and observability overlap but are not the same. **Monitoring** watches for failures you already know about: a dashboard of CPU, an alert when the error rate crosses 1%. **Observability** lets you ask questions you did not think of in advance: "Are slow checkouts all from users on the new Android build who pay with one particular card issuer?" Monitoring tells you a known thing is wrong. Observability lets you investigate an unknown thing.

In a single-process application you could attach a debugger or read one log file. In a distributed system a single user click might touch a gateway, five services, two databases, a cache and a queue, running on dozens of machines. No one machine holds the whole story. The data each component emits about itself is called **telemetry**, and it comes in three main signals: **metrics**, **logs** and **traces**. Each answers a different kind of question, and a real investigation usually moves between all three.

~~~dg-matrix Each signal is good at a different question; real debugging moves between all three.
Signal | Is something wrong? | Which service? | Exactly why? | Cheap at scale?
Metrics | yes, alerts and dashboards | partial, per-service graphs | no, aggregated away | yes, fixed cost per series
Traces | partial, if sampled well | yes, the waterfall shows it | partial, span attributes | partial, needs sampling
Logs | partial, by counting lines | partial, if trace IDs present | yes, full event detail | no, cost grows with traffic
~~~

## Metrics: is something wrong, and how much?

A **metric** is a number aggregated over time: requests per second, error count, CPU utilisation, queue depth, latency percentiles. A metric is stored as a **time series**: a name, a set of labels (for example `service="checkout", region="eu-west"`) and a stream of (timestamp, value) points. A million requests become a single counter incremented a million times. That is why metrics are cheap: their storage cost depends on how many series you have, not on how much traffic you serve.

That fixed cost has one trap. Every unique combination of label values creates a new series. A label like `user_id` with 10 million values creates 10 million series, and the metrics database falls over. This is the **cardinality** problem. Keep metric labels to low-cardinality values (service, endpoint, status code, region) and put high-cardinality detail in logs and traces.

The weakness of metrics is that they throw detail away. A latency graph can show checkout p99 jumping from 300 ms to 2 s. It cannot show you which request, which user or which downstream call was slow.

### The four golden signals

If you can measure only four things about a service, measure these:

- **Latency**: time to serve a request. Track successful and failed requests separately. A fast HTTP 500 drags the average down and hides the problem, and a slow error is worse than a fast one.
- **Traffic**: demand in the system's own unit. HTTP requests per second for a web service, concurrent sessions for a video stream, reads and writes per second for a key-value store.
- **Errors**: failed requests. They can be explicit (HTTP 500), implicit (HTTP 200 with the wrong content) or by policy (anything slower than a promised 1 s counts as failed). Load balancer counts catch outright failures; only end-to-end probes catch wrong content.
- **Saturation**: how full the most constrained resource is (CPU, memory, connection pool, disk, queue). Many systems degrade well before 100%, so pick a target utilisation. Rising latency is often the first sign of saturation, and predictions count too ("the disk fills in 4 hours").

~~~dg-layers The four golden signals, from what users feel at the top to what predicts the next outage at the bottom.
*Latency | how long requests take; split success and failure
Traffic | how much demand: requests/s, sessions, transactions
Errors | explicit 500s, wrong 200s, too slow counts as failed
Saturation | how full the tightest resource is
~~~

Two related shorthands are worth knowing. **RED** (Rate, Errors, Duration) is the golden signals minus saturation, used for request-driven services. **USE** (Utilisation, Saturation, Errors) is for resources such as disks, CPUs and network links. Use RED on every service endpoint and USE on every machine and pool.

## Why averages lie

A service handling 1,000 requests per second with a 100 ms **average** latency can still have 1% of requests (10 every second) taking 5 seconds. The average hides the tail, and the tail is what users complain about. It gets worse with fan-out. If one page calls 20 backends in parallel and each has a 1% chance of being slow, the chance that at least one is slow is 1 − 0.99^20, about 18%. One backend's 99th percentile becomes the page's typical case.

The fix is to record **histograms**: count each request into a latency bucket instead of keeping raw values or a single mean. Buckets usually grow geometrically, roughly ×2 to ×3 each step (0 to 10 ms, 10 to 30, 30 to 100, 100 to 300, 300 ms to 1 s, 1 to 3 s). From the bucket counts you can estimate any percentile and see the shape of the distribution. Crucially, histograms can be **added together**: you can merge histograms from 200 servers and compute a correct fleet-wide p99. You cannot average 200 per-server p99 values and get the fleet p99; that number is meaningless.

~~~dg-bars Latency of one checkout endpoint over five minutes. The mean looks healthy; the tail is where users suffer.
p50 | 120 | 120 ms
Mean | 180 | 180 ms
p90 | 310 | 310 ms
*p99 | 2000 | 2,000 ms
p99.9 | 4800 | 4,800 ms
~~~

Resolution should match the goal. A CPU graph averaged per minute hides one-second spikes that cause tail latency. A useful trick: sample CPU every second into a histogram with 5% buckets, and ship that histogram once a minute. You keep the spikes without paying for per-second storage.

<div class="ex">
<div class="lab">Worked example</div>

**sizing a metrics system**

You run 300 services, each on about 20 pods. Each pod exports 150 series (request counts per endpoint and status, a latency histogram with 12 buckets per endpoint, runtime stats). That is 300 × 20 × 150 = 900,000 active series.

Scraping every 15 seconds gives 900,000 ÷ 15 = 60,000 samples per second. Compressed time-series storage needs about 1 to 2 bytes per sample, so a day is 60,000 × 86,400 × 2 ≈ 10 GB. Very manageable.

Now a developer adds a `customer_id` label to the request counter, with 50,000 active customers. One metric explodes to 300 × 20 × 50,000 = 300 million series. The metrics database runs out of memory within the hour. That one label is the classic cardinality outage.

</div>

## Logs: what exactly happened?

A **log** is a timestamped record of a single event. Logs carry the detail metrics throw away: the user ID, the full error message, the SQL that timed out, the version of the client. Their cost grows with traffic: at 20,000 requests per second, with 10 log lines of 500 bytes each per request, you write 100 MB per second, about 8.6 TB per day. That is why teams sample debug logs, keep only a few days of hot storage and drop noisy lines.

Logs on their own **lack context**. A line that says "payment call failed" does not say which incoming request triggered it or what else that request did on other machines. Two changes make logs useful in a distributed system:

- Make them **structured**: emit JSON fields (`{"level":"error","order_id":"o-981","status":503,"trace_id":"abc"}`) instead of free text, so you can filter and group by any field rather than grep for phrases.
- Put a **trace ID** (or request ID) in every line, so you can pull every log line for one request across every service with a single query.

## Traces: where did the time go?

A **span** is one unit of work: an HTTP handler, a database query, a call to another service. A span has a name, a start time, a duration, a parent span ID, **attributes** (key-value metadata such as `http.request.method = GET`, `http.route = /webshop/articles/:article_id`, `http.response.status_code = 200`) and span events (structured log messages inside it).

A **trace** is the tree of spans for one request. The root span covers the request from start to finish; child spans cover the steps inside it. Tools draw a trace as a **waterfall**: one horizontal bar per span, nested under its parent and placed on a shared time axis, so the long bar is obvious at a glance. Gaps between child spans are also informative: they show time spent inside the service itself (queuing, garbage collection, CPU work) rather than waiting on a dependency.

### Context propagation

For a trace to cross service boundaries, every service must pass the trace ID and its own span ID along with each outgoing call. Over HTTP this is a header; the W3C standard is `traceparent`, which looks like `00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01`: a version, a 16-byte trace ID, an 8-byte parent span ID and flags (the last bit says whether this trace is sampled). The receiving service reads the header, creates a child span with that parent, and passes its own span ID onward. Over a message queue, the same context travels in message headers, so a trace can follow work into an asynchronous consumer.

Propagation is the fragile part. One service that forgets to forward the header, or one thread pool that drops the context, splits the trace in two. Use an instrumentation library (OpenTelemetry is the common standard) that patches HTTP clients, database drivers and queue clients automatically.

~~~dg-seq Context propagation: each hop reads the incoming trace context, starts a child span and forwards the context on.
Browser > API gateway : POST /checkout (no trace header)
API gateway > API gateway : start trace abc, root span g1
API gateway > Checkout : traceparent abc / parent g1
Checkout > Checkout : start span c1, parent g1
Checkout > Payment : traceparent abc / parent c1
Payment > Payment : start span p1, parent c1 (1,700 ms)
Payment -> Checkout : 200 OK after 1.7 s
Checkout -> API gateway : 200 OK after 1.85 s
API gateway -> Browser : 200 OK after 1.9 s
~~~

### Sampling

Recording every span of every request is expensive at high traffic. **Head-based sampling** decides at the start of the trace (for example, keep 1% at random) and propagates the decision in the flags, so all services agree. It is cheap but will usually miss the rare slow or failed request you care about. **Tail-based sampling** buffers all spans of a trace in a collector until it completes, then keeps it if it was slow, failed or otherwise interesting, plus a small random share of normal traces. It catches the outliers but needs a collector tier with enough memory to hold every in-flight trace for some seconds.

<div class="ex">
<div class="lab">Worked example</div>

**debugging a slow checkout end to end**

1. **Metrics** raise the alert: checkout p99 rose from 300 ms to 2 s at 14:05, while the error rate is flat. Traffic is normal, so it is not simply load.

2. **Traces**: open a handful of slow checkout traces from after 14:05. In the waterfall the `payment.authorize` span takes 1.7 s of the 1.9 s; inventory and pricing spans are normal. The slow service is the payment service.

3. **Logs**: filter payment-service logs by those trace IDs. They show "waiting for connection from pool" for 1.6 s before each call to the card processor.

4. Back to **metrics** to confirm the cause: the payment service's connection pool saturation is at 100% since a 14:00 deploy that halved the pool size from 40 to 20. Roll back, and p99 returns to 300 ms within two minutes.

</div>

## Alerting on symptoms, not causes

Separate **what is broken** (the symptom: errors, slow responses users feel) from **why** (the cause: database refusing connections, CPU at 100%, a failed disk). Page people on symptoms. Use cause metrics on dashboards for debugging, and page on a cause only when it is definite and imminent, such as a disk that will fill within an hour or a cluster down to its last replica.

Why? A cause alert can fire with no user impact (CPU at 95% may be fine), and many causes can produce one symptom, so you would need a cause alert for every possible failure and still miss the new one. A symptom alert catches every failure that matters, including ones you never imagined.

Rules for pages:

- Every page must be **urgent**, **actionable** and need **human judgement**. If the response is mechanical (restart the process, add a node), automate it instead.
- People can only respond with urgency a few times per shift before fatigue sets in. Noisy alerts train people to ignore pages, and then they miss the real one.
- Delete alert rules that rarely fire usefully, and signals that appear in no dashboard and no alert.

The Bigtable team at Google learned the cost of noise. Their SLO was set on mean latency, but the slowest 5% of requests were much slower, so alerts fired constantly and engineers missed real problems among the noise. The team temporarily set the target at the 75th percentile and turned off email alerts. With fewer interruptions, the on-call engineers had the time to fix the underlying causes, and the target was tightened again later.

## Common mistakes

- **Averages on dashboards.** Plot p50, p99 and p99.9 from histograms. Never average percentiles across hosts.
- **High-cardinality labels on metrics.** User IDs, order IDs and URLs with IDs in them belong in logs and traces.
- **Unstructured logs with no trace ID.** You end up grepping 40 machines and guessing which lines belong together.
- **Broken propagation.** One service or async hop that drops the trace header, so traces end abruptly at that boundary.
- **Random head sampling only.** At 1% you discard 99% of the rare failures you most want to see.
- **Alerting on every cause.** Dozens of CPU and memory alerts with no user impact, and on-call learns to ignore them.

## In an interview

When you finish a design, add an observability paragraph. Say you would emit RED metrics per endpoint as histograms, structured logs with trace IDs, and distributed traces with OpenTelemetry and tail-based sampling. Say you would alert on user-facing SLOs (lesson 14.02 covers burn-rate alerts) and keep cause metrics for dashboards. If the interviewer asks "checkout is slow, how do you find out why?", walk the loop: metric alert, slow traces, span waterfall, logs for that span by trace ID, confirm with a cause metric.

<div class="take">
<div class="lab">Key takeaway</div>

metrics tell you something is wrong and how much; traces tell you which service and which call; logs tell you exactly what happened inside it. Record latency as histograms, keep metric labels low-cardinality, propagate a trace ID through every hop, and page only on symptoms users feel.

</div>
