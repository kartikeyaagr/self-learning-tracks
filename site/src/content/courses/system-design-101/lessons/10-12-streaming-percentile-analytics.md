---
id: "10.12"
course: "system-design-101"
module: "10"
slug: "streaming-percentile-analytics"
title: "Streaming percentile analytics"
keyIdeas: ["Percentiles don't average: the average of 10 servers' p99s is not the global p99. One server might have all the slow requests.", "To get correct global percentiles, aggregate the distributions, not the summary numbers: merge histograms, t-digests or other sketches.", "Prometheus histograms count observations into fixed buckets; percentiles are estimated from the merged bucket counts, so bucket boundaries should be near the thresholds you care about.", "Prometheus summaries compute quantiles per instance and can't be aggregated across instances."]
question: "Why can't you average p99s from 10 servers to get the global p99?"
answer: "A p99 is a single point of each server's distribution, not something that adds. Servers handle different amounts of traffic, and the global p99 is determined by where the slowest 1% of all requests fall, which may be almost entirely on one degraded server: with nine servers at 100 ms and one at 2 s handling a tenth of traffic, the average of p99s is 290 ms while the true global p99 is close to 2 s. No formula recovers the global value from ten percentiles, because the rest of each distribution has been thrown away. Instead, ship additive data such as histogram bucket counts or mergeable sketches like t-digest, sum or merge them across servers, and compute the global p99 once from the combined distribution."
sources: [{"title": "Prometheus: Histograms and summaries", "url": "https://prometheus.io/docs/practices/histograms/", "note": "Percentiles in monitoring."}]
---

## The trap: averaging percentiles

Ten servers each report their p99 latency. The tempting global p99 is the average of the ten numbers. Dashboards do this every day, often without anyone noticing, because the query looks reasonable: `avg(http_request_duration_seconds{quantile="0.95"})` in Prometheus, or "average of p99" in a metrics UI. It is wrong. Averaging quantiles gives **statistically nonsensical values**: numbers that don't correspond to any real percentile of your traffic.

Why it is wrong:

- **Servers carry different traffic.** One server handling 90% of requests and another handling 1% count equally in an average.
- **Quantiles aren't linear.** The global p99 depends on where the slowest 1% of **all** requests lie. If one server is degraded, nearly all of the global slowest 1% may come from it; its p99 is high, the other nine are low, and their average hides the problem.
- **There is no formula** from ten p99 values to the true global p99. The information about the rest of each distribution is gone, the same non-additivity as distinct counts in lesson 10.05. Even a traffic-weighted average of p99s is wrong, because percentiles are not linear in the data.

<div class="ex">
<div class="lab">Worked example</div>

**the average hides a bad server**

Ten servers each handle 10,000 requests a minute, so 100,000 in total. Nine are healthy with p99 = 100 ms. The tenth has a slow disk: 20% of its requests take about 2 seconds, so its p99 is about 2,000 ms.

Average of the ten p99s: (9 × 100 + 2,000) / 10 = 290 ms. Looks like a mild problem.

True global p99: the slowest 1% of all requests is the slowest 1,000. The bad server alone has 2,000 requests at about 2 seconds, so all 1,000 of the global slowest are among them. The global p99 is about 1,900 to 2,000 ms. One user in a hundred waits two seconds, and the averaged dashboard says 290 ms.

</div>

~~~dg-bars The same traffic summarised two ways. Averaging the servers' p99s reports a number that matches no real percentile of the traffic.
9 healthy servers, p99 each | 100 | 100 ms
Bad server p99 | 2000 | 2,000 ms
Average of 10 p99s | 290 | 290 ms
*True global p99 | 1950 | about 1,950 ms
~~~

<div class="ex">
<div class="lab">Worked example</div>

**the average can also cry wolf**

Now flip it. Nine servers handle 99% of traffic with p99 = 100 ms. A tenth is a canary handling 1% of traffic, and its p99 is 5,000 ms because it is cold.

Average of p99s: (9 × 100 + 5,000) / 10 = 590 ms, which triggers an alert. True global p99: the canary's 1% of traffic includes maybe 1% slow requests, which is 0.01% of all traffic, nowhere near the slowest 1%. The global p99 is still about 100 ms. The average is wrong in both directions.

</div>

## What to aggregate instead: things that add

The fix is a general principle: **ship additive data, combine it, then compute the quantile once.** Counts add. Sums add. Bucket counts add. Sketches merge. Percentiles do neither.

Prometheus offers two metric types for latency, and the difference is exactly this:

| | Histogram | Summary |
| --- | --- | --- |
| What the client exports | Counts per bucket (`le` = less than or equal), plus `_sum` and `_count` | Pre-computed quantiles, plus `_sum` and `_count` |
| Client cost | Cheap counter increments | Relatively expensive streaming quantile calculation |
| Quantile computed | On the server, at query time, any quantile | In the client, fixed quantiles and window chosen up front |
| Error | Limited by bucket width | Configurable, usually very low, but per instance |
| Aggregation across instances | Yes: bucket counts add | **No** |

A histogram bucket `http_request_duration_seconds_bucket{le="0.3"}` is a counter of how many requests took 0.3 seconds or less since the process started. It is cumulative: the `le="0.5"` bucket includes everything in `le="0.3"`. Because each bucket is a plain counter, the same bucket on 40 servers can be summed to get the fleet-wide count.

With histograms, the global p95 across replicas is:

`histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket[5m])))`

Read it inside out. `rate(...[5m])` turns each server's ever-growing counters into requests per second over the last 5 minutes, per bucket. `sum by (le)` adds the same bucket across every server, giving one fleet-wide histogram. Only then does `histogram_quantile` estimate the 0.95 quantile from it.

~~~dg-flow Correct aggregation: sum the additive bucket counts across servers first, then compute the percentile once from the combined histogram.
10 servers: bucket counts -(rate, then sum by le)-> *Global histogram -(histogram_quantile 0.99)-> Global p99
~~~

~~~dg-matrix What you can do with each kind of latency data once it leaves the server.
Data shipped | Sum across servers | Roll up over time | Any quantile later | Cost per request
Raw latencies | yes | yes | yes | huge, store everything
Per-server p99 | no | no | no | tiny
Histogram buckets | yes | yes | partial, bucket-limited | tiny
t-digest or DDSketch | yes, merge | yes, merge | yes | small
Mean only | partial, weight by count | partial | no | tiny
~~~

## How histogram_quantile estimates

`histogram_quantile` finds the bucket containing the target rank and **interpolates linearly** inside it, assuming values are spread evenly within the bucket.

<div class="ex">
<div class="lab">Worked example</div>

**computing p95 from buckets by hand**

The summed per-second rates are: le 0.1 = 9,000; le 0.2 = 9,600; le 0.3 = 9,950; le +Inf = 10,000. So 10,000 requests per second in total.

p95 is rank 0.95 × 10,000 = 9,500. That falls in the bucket from 0.1 to 0.2 seconds, which holds 9,600 − 9,000 = 600 requests per second, and rank 9,500 is 500 into it.

Linear interpolation: 0.1 + (500 / 600) × (0.2 − 0.1) = 0.183 seconds. Prometheus reports p95 = 183 ms. The real p95 could be anywhere between 100 and 200 ms; 183 assumes the 600 requests are evenly spread across that bucket.

</div>

## The price of buckets

So the error is bounded by the bucket's width, and badly placed buckets produce misleading answers. An example with an SLO of p95 under 300 ms:

- Requests actually cluster at **220 ms**. A classic histogram with a 200 to 300 ms bucket estimates p95 as **295 ms**, making a healthy service look close to breaching its SLO.
- A **native histogram** (fine exponential buckets, each about 1.09 times the last) puts the value in a 210 to 229 ms bucket and estimates **228 ms**.
- After a change shifts latencies to 320 ms, the native histogram says 323 ms, while a classic 300 to 450 ms bucket says 443 ms.

~~~dg-bars Estimated p95 for requests that really cluster at 220 ms: wide classic buckets push the estimate towards the bucket's upper edge, while fine exponential buckets stay close.
True latency | 220 | 220 ms
*Native histogram estimate | 228 | 228 ms
Classic bucket 200 to 300 | 295 | 295 ms
~~~

Hence the advice: choose bucket boundaries around your SLO thresholds (for example a boundary at exactly 0.3 s), or use native histograms, whose exponential buckets give bounded **relative** error everywhere. OpenTelemetry's exponential histograms map directly onto them. With a boundary at the SLO, the question that matters most, "what fraction of requests met the SLO?", is answered exactly: it is `le="0.3"` divided by the total, with no interpolation.

Each bucket is a separate time series per label combination, so buckets aren't free. 12 buckets × 40 pods × 20 endpoints × 5 status codes is 48,000 series for one metric. Native histograms store the populated buckets compactly in one series, which is another reason they exist.

Averages are still easy: `rate(_sum) / rate(_count)` gives mean latency, and both series add across instances. The mean is the one statistic you can combine correctly from per-server data, as long as you combine the sums and counts rather than averaging per-server means.

## Other mergeable options

The same principle covers the sketches in this module: ship a **t-digest** (lesson 10.11), **DDSketch** or **HdrHistogram** per server per interval, merge them centrally, then read any quantile. DDSketch uses logarithmically sized buckets so every value is within a chosen relative error (say 1%) and merging is just adding bucket counts. HdrHistogram does the same over a fixed value range with configurable precision. Prometheus histograms are the simplest bucketed version of this idea; native and OpenTelemetry exponential histograms are essentially DDSketch-style buckets.

### Time matters too

The same trap appears across time. "Average the p99 of each minute to get the hour's p99" is wrong for the same reason: a bad minute with 10 times the traffic counts the same as a quiet one. Store histograms or sketches per minute and merge them for the hour.

## Common mistakes

- **`avg()` over a quantile label.** Always wrong for combining instances; at best a rough "typical instance" view.
- **Using Summaries for anything you'll aggregate.** Summaries are fine for a single process; they can't be combined across replicas.
- **Default buckets.** Library defaults (5 ms to 10 s, roughly doubling) rarely match your SLO; add a boundary at the threshold you alert on.
- **`sum` without `by (le)`.** Summing away the `le` label merges all buckets into one number and destroys the histogram.
- **Forgetting `rate` before `sum`.** Summing raw cumulative counters across pods that restarted at different times gives garbage; take the rate per series first.

<div class="ex">
<div class="lab">Worked example</div>

**an SLO dashboard**

Each of 40 API pods exports `http_request_duration_seconds_bucket` with boundaries 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2.5 (0.3 deliberately included because the SLO is p95 under 300 ms).

Global p99: `histogram_quantile(0.99, sum by (le) (rate(http_request_duration_seconds_bucket[5m])))`.

To spot a bad pod: the same query with `sum by (le, pod)`, which gives one p99 per pod. The fleet line and the pod lines are both correct; only averaging the pod lines would be wrong.

SLO compliance: `sum(rate(..._bucket{le="0.3"}[5m])) / sum(rate(..._count[5m]))`, the fraction of requests under 300 ms. It is exact, because 0.3 is a bucket boundary: if it reads 0.962, then 96.2% of requests met the target and the SLO (95%) is met.

</div>

## In an interview

When you add monitoring to a design, say "each service exports latency histograms (or sketches); we aggregate bucket counts across instances and compute percentiles centrally, never averaging per-instance percentiles". If asked why, give the bad-server example in one sentence: the global slowest 1% can all come from one server, so the average of ten p99s can be several times too low.

<div class="take">
<div class="lab">Key takeaway</div>

percentiles don't average, because each server's p99 hides its traffic volume and the shape of its distribution. Aggregate additive data instead: sum histogram bucket counts (or merge sketches such as t-digest or DDSketch) across servers and across time, then compute the quantile once. Bucket placement determines accuracy, so put boundaries at your SLO thresholds or use native exponential histograms.

</div>
