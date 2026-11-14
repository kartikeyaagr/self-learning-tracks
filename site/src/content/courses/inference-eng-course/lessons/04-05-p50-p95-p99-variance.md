---
id: "04.05"
course: "inference-eng-course"
module: "04"
slug: "p50-p95-p99-variance"
title: "P50, P95, P99 & Variance"
keyIdeas: ["Means hide tails. Report p50, p95 and p99 for each latency metric.", "Percentiles need enough samples: thousands for a stable p99.", "Do not average percentiles across runs or servers; merge raw samples or use histograms.", "Coordinated omission: a closed-loop client that waits can hide stalls. Open-loop timing avoids it.", "Look at distributions (histograms, CDFs), not just three numbers."]
question: "Your p50 TTFT improved by 20% after enabling a feature, but p99 got 40% worse. Should you ship it? What would you check?"
answer: "It depends on the SLO, which is usually written against a tail percentile, so the worse p99 probably fails it. Check which requests became slow (long prompts? specific arrival bursts?), whether the change is reproducible across runs, and whether the p99 is measured on enough samples. Often a scheduling change helps typical requests at the expense of long ones."
sources: [{"title": "The Tail at Scale", "url": "https://research.google/pubs/the-tail-at-scale/", "note": "Why tails dominate user experience at scale."}, {"title": "Gil Tene: How NOT to Measure Latency (talk)", "url": "https://www.youtube.com/watch?v=lJ8ydIuPFeU", "note": "Coordinated omission and percentile pitfalls."}]
tags: ["statistics"]
---

## Why averages hide the problem

A mean TTFT of 200 ms could mean every request took 200 ms. It could also mean 95% took 100 ms and 5% took 2.1 s. The second case has the same mean and a much worse product, because one user in twenty waits two seconds. Latency distributions in LLM serving are skewed: most requests are fast, and a tail of slow ones comes from queueing, long prefills, cache misses and preemption.

That is why SLOs are written in percentiles, such as "p95 TTFT under 500 ms" or "p99 ITL under 200 ms".

## What a percentile means

The p95 is the value below which 95% of samples fall. With 1,000 requests, it is roughly the 950th smallest value. The p50 is the median. The p99 is the value that only 1% of requests exceed.

~~~dg-raw
%%widget pct Raise the spread and watch p99 pull away from the median while the median barely moves.
m = 400
s = 0.6
%%
~~~

## How many samples you need

A percentile estimate is only as good as the number of samples above it.

- With 10,000 requests, the p99 rests on the 100 slowest. That is reasonably stable.
- With 1,000 requests, it rests on 10. Expect it to move by 20% or more between runs.
- With 300 requests, it rests on 3. It is barely more than the third-worst request.

A rough rule: to estimate the p-th percentile, you want at least 100 samples beyond it. For p99 that means at least 10,000 requests. For p99.9, 100,000.

## Aggregation mistakes

- **Averaging percentiles.** If server A has p99 of 300 ms and server B has p99 of 900 ms, the p99 of their combined traffic is not 600 ms. It depends on the full distributions. Merge raw samples, or use mergeable histograms (such as Prometheus histograms or HDR histograms), and compute the percentile from the merged data.
- **Percentiles of per-request averages.** TPOT is already an average within each request. Its p99 across requests is not the same as the p99 of all individual token gaps. Report ITL percentiles for stalls and TPOT percentiles for overall streaming speed.
- **Weighting.** A median over requests treats a 5-token answer the same as a 2,000-token one. If you care about the user experience per token, compute percentiles over tokens.

## Coordinated omission

This is the most important benchmarking mistake to understand. Suppose a closed-loop client sends one request, waits for the response, then sends the next. The server stalls for 5 seconds. During the stall, the client sends nothing. When the stall ends, it resumes, and its next requests record normal latency.

In reality, users would have kept arriving during those 5 seconds, and every one of them would have waited. The client "coordinated" with the server and omitted exactly the samples that would have shown the problem.

<div class="ex">
<div class="lab">Worked example</div>

**coordinated omission in numbers**

An open-loop client sends 10 requests per second. Normally each takes 200 ms. A 5 s stall happens once in a 100 s test.

Correct measurement: about 50 requests arrive during the stall, waiting 2.5 s on average and up to 5 s. With 1,000 requests in total, p95 is about 1.8 s.

A closed-loop client with one user records only one slow request (5.2 s) among about 450. Its p95 is 200 ms, and the stall is invisible.

</div>

The fix: send requests on a schedule set by the clock, not by completions, and measure latency from the scheduled send time.

## Look at the whole distribution

Percentiles summarize, but a plot of the full distribution often tells you the cause. Plot the cumulative distribution (CDF) of TTFT and look for its shape:

- **One smooth curve**: variance from normal queueing.
- **Two steps**: two populations, such as prefix cache hits and misses, or short and long prompts. Fix each separately.
- **A long flat tail**: occasional large stalls, such as preemption, garbage collection or a long prefill blocking others.

| Metric | p50 | p95 | p99 |
| --- | --- | --- | --- |
| TTFT (ms) | 120 | 410 | 1,900 |
| ITL (ms) | 22 | 31 | 240 |

In this table, p99 ITL is about ten times p50. That pattern usually means occasional stalls, often from long prefills running beside decodes. Chunked prefill is the usual fix.

## Deep dive: how good is a p99 from n samples?

The estimated p99 is the sample at rank ⟦m400⟧. The number of samples below the true p99 is binomial with mean ⟦m401⟧ and standard deviation ⟦m402⟧. So with 95% confidence, the estimate lands between the true

⟦M398⟧

percentiles.

~~~dg-raw
%%chart Width of the 95% band around a p99 estimate, in percentile points (±).
x = 100, 300, 1000, 3000, 10000, 100000
logx = 1
logy = 1
xlab = Samples
ylab = ± percentile points
p99 estimate: 1.95, 1.13, 0.62, 0.36, 0.2, 0.06
p999 estimate: 0.62, 0.36, 0.2, 0.11, 0.06, 0.02
%%
~~~

<div class="ex">
<div class="lab">Worked example</div>

**why 1,000 requests is too few for p99**

With 1,000 samples, the reported p99 is really somewhere between p98.4 and p99.6. In a latency distribution with a heavy tail, p98.4 might be 900 ms and p99.6 might be 3 s.

The rule of thumb is to keep at least 100 samples beyond the percentile you report: 10,000 requests for p99 and 100,000 for p99.9.

</div>

## Deep dive: percentiles do not average

For two servers, the p99 of the merged traffic is not the mean of their p99s. If server A has 1,000 requests with p99 = 200 ms and server B has 1,000 with p99 = 2,000 ms, the merged p99 is the 20th-slowest of 2,000. Nearly all of those come from B, so the answer sits near B's p98. That could be anywhere from 1 s to 2 s. The mean of the two p99s, 1,100 ms, has no meaning. Keep histograms, which can be added bin by bin, and compute percentiles from the sum.

## Deep dive: coordinated omission, formally

Let requests be scheduled at ⟦m403⟧. The correct latency of request ⟦m404⟧ counts from its schedule, not from when it was sent:

⟦M399⟧

If the server stalls for ⟦m405⟧ seconds, about ⟦m406⟧ scheduled requests fall inside the stall, with latencies spread roughly evenly from 0 to ⟦m407⟧. Their mean is about ⟦m408⟧. A client that waits for each reply before sending the next records one of them and silently drops the other ⟦m409⟧. The tail you lose is the very tail you were trying to measure.

<div class="take">
<div class="lab">Key takeaway</div>

report tail percentiles with enough samples behind them, merge raw data rather than percentiles, avoid coordinated omission by sending on the clock, and plot the distribution to find what causes the tail.

</div>
