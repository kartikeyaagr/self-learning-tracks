---
id: "06.04"
course: "inference-eng-course"
module: "06"
slug: "read-engine-metrics-request-state"
title: "Read Engine Metrics & Request State"
keyIdeas: ["Engines expose Prometheus metrics at /metrics.", "Running vs waiting requests: queue pressure.", "KV cache usage percent: memory pressure.", "Preemptions: memory pressure bad enough to evict running work.", "Histograms of TTFT, TPOT, E2E and prompt/generation lengths from the server's view."]
question: "During a load test you see num_requests_waiting rising and KV cache usage at 99%. What is happening and what are two fixes?"
answer: "The KV cache is full, so the scheduler cannot admit new sequences and they queue, raising TTFT. Fixes: free KV space (quantize the KV cache to FP8, quantize weights, raise gpu-memory-utilization, lower max-model-len if long requests reserve too much), or add capacity (another replica) and route traffic away; also check if requests are generating longer outputs than expected."
sources: [{"title": "vLLM: Metrics", "url": "https://docs.vllm.ai/en/latest/usage/metrics.html", "note": "Prometheus metrics exported by vLLM."}, {"title": "Prometheus: Overview", "url": "https://prometheus.io/docs/introduction/overview/", "note": "Scraping and querying metrics."}]
tags: ["metrics", "vllm"]
---

## The engine tells you why

Client-side benchmarks tell you what happened: latency went up. The engine's own metrics tell you why: requests were queueing, the cache was full, or sequences were being preempted. Scrape both during every test and every production hour.

## Where to look

vLLM and SGLang serve Prometheus-format metrics at `GET /metrics`. Scrape every 5 to 15 seconds with Prometheus, or poll it with a small script during a benchmark.

## The metrics that matter most

| Metric (names vary by engine and version) | Meaning |
| --- | --- |
| num_requests_running | sequences in the current batch |
| num_requests_waiting | accepted by the engine but not yet scheduled |
| kv_cache_usage_perc | fraction of KV blocks in use |
| num_preemptions_total | sequences evicted for lack of memory |
| prefix_cache_hits and queries | share of prompt tokens served from the cache |
| time_to_first_token_seconds | server-side TTFT histogram |
| time_per_output_token_seconds | server-side TPOT histogram |
| e2e_request_latency_seconds | server-side total latency histogram |
| prompt_tokens_total, generation_tokens_total | throughput counters |
| request_success_total, by finish reason | how requests ended |

Counters such as `generation_tokens_total` only ever increase. To get throughput, take the rate over a window, such as `rate(generation_tokens_total[1m])`.

## Request states

Inside the engine, each request moves through a small set of states:

~~~dg-flow A request's lifecycle inside the engine. Press Play to step through it.
Waiting -(scheduled)-> *Running -(KV full)-> Preempted -(rescheduled)-> Running -> Finished
~~~

- **Waiting**: in the engine's queue. Its TTFT is growing.
- **Running**: in the batch. Each step it either prefills a chunk of its prompt or decodes one token.
- **Preempted**: removed from the batch to free memory. It goes back to waiting and will recompute or swap in later.
- **Finished**: stopped by a stop token, a length limit, an abort or an error.

## Reading the patterns

| Pattern | Likely cause | What to do |
| --- | --- | --- |
| Waiting grows, KV usage near 100% | memory bound | FP8 KV, more memory per replica, more replicas |
| Waiting grows, KV usage low | max-num-seqs cap or compute bound | raise the cap if latency allows, or add replicas |
| Preemptions rising | memory overcommitted | lower the concurrency cap, add capacity |
| Server TTFT much lower than client TTFT | time lost before the engine | check the gateway, proxy buffering, client pools |
| Running high, throughput flat | compute saturated | more GPUs, or a smaller or quantized model |
| Prefix hit rate low on a shared-prompt product | prompts differ in the prefix | move variable content after the shared part |

<div class="ex">
<div class="lab">Worked example</div>

**diagnosing a latency rise from metrics**

At 14:05, p95 TTFT on the dashboard jumps from 300 ms to 2.8 s. TPOT stays at 22 ms.

Engine metrics: running steady at 128, waiting rises from 0 to 70, KV usage at 61%.

KV is not full, so memory is not the limit. Running is stuck at exactly 128, the configured max-num-seqs.

Diagnosis: the sequence cap, not the hardware, is limiting admission. Raising the cap to 192 brings TTFT back under 400 ms while TPOT rises to 27 ms, still within the 40 ms SLO.

</div>

## Building a dashboard

A useful first dashboard has six panels: request rate and error rate, p50 and p95 TTFT, p50 and p95 TPOT, running and waiting requests, KV cache usage, and output tokens per second. Add the preemption rate and the prefix cache hit rate as soon as you use those features. Put the client-side TTFT next to the server-side one, so the gap between them is visible.

## Deep dive: what a Prometheus histogram can and cannot tell you

Latency histograms store counts in fixed buckets. `histogram_quantile` finds the bucket holding the quantile and interpolates linearly inside it:

⟦M474⟧

with bucket bounds ⟦m476⟧, cumulative counts ⟦m477⟧, and target quantile ⟦m478⟧. The true value can be anywhere in ⟦m479⟧.

<div class="ex">
<div class="lab">Worked example</div>

**a p99 that is really a range**

TTFT buckets are 0.1, 0.25, 0.5, 1, 2.5 and 5 s. Out of 10,000 requests, 9,850 are under 0.5 s and 9,960 under 1 s.

The p99 (rank 9,900) is in the 0.5 to 1 s bucket. Interpolation gives ⟦m480⟧ s.

The true p99 could be anywhere from 0.5 to 1 s. If the SLO is 800 ms, this histogram cannot tell you whether you meet it. Add a bucket at 0.8.

</div>

## Deep dive: Little's law on engine gauges

Engines export the number running, the number waiting, and a counter of finished requests. Their ratio gives the mean time in the engine without any client-side timing:

⟦M475⟧

If the client reports a mean E2E much larger than ⟦m481⟧, the missing time is outside the engine: the load balancer, the API gateway, or the client's own connection pool.

<div class="take">
<div class="lab">Key takeaway</div>

the engine's metrics tell you why latency rose. Check the waiting queue, KV usage and preemptions first, compare server and client TTFT, and read counters as rates.

</div>
