---
id: "23.02"
course: "inference-eng-course"
module: "23"
slug: "queue-aware-routing"
title: "Queue-Aware Routing"
keyIdeas: ["Route using each replica's queue depth, running sequences and KV usage.", "Metrics are stale by seconds; routers must account for requests in flight.", "Herding: everyone picks the same \"best\" replica.", "Power-of-two-choices gives most of the benefit with stale data.", "Estimate request cost from prompt length (known) and expected output length."]
question: "What is \"herding\" in queue-aware routing and how do you prevent it?"
answer: "When many routers see the same replica as least loaded (from metrics a few seconds old), they all send to it at once, overloading it while others go idle; the pattern then flips. Prevent it with power-of-two-choices (sample two replicas at random, pick the better), local tracking of requests each router has just sent, and short metric staleness windows."
sources: [{"title": "The Power of Two Choices in Randomized Load Balancing", "url": "https://www.eecs.harvard.edu/~michaelm/postscripts/mythesis.pdf", "note": "Mitzenmacher's analysis."}, {"title": "vLLM: Metrics", "url": "https://docs.vllm.ai/en/latest/usage/metrics.html", "note": "Queue and KV metrics to route on."}]
tags: ["routing"]
---

## Signals

| Signal | Source |
| --- | --- |
| Waiting requests | engine metrics endpoint |
| Running sequences | engine metrics endpoint |
| KV cache usage | engine metrics endpoint |
| Requests sent recently | the router's own count |
| Prompt tokens of pending requests | the router knows them |

## A simple score

```
score = waiting_tokens + alpha * running_seqs + beta * kv_usage
pick  = min(random.sample(replicas, 2), key=score)
```

- **waiting_tokens** predicts how long a new request will wait for prefill.
- **running_seqs** predicts how slow each decode step is.
- **kv_usage** warns that the replica is close to preempting or queueing requests.

Set alpha and beta by replaying a trace and measuring TTFT p95. They depend on the model and hardware.

## Herding

Metrics are typically scraped every 1 to 5 seconds. With many routers acting on the same stale numbers, they all pick the same replica, overload it, then all move to the next one together.

~~~dg-seq Herding with stale metrics.
Router 1 > Replica B : B looks idle
Router 2 > Replica B : B looks idle
Router 3 > Replica B : B looks idle
Replica B > Replica B : queue jumps
~~~

Mitigations:

- **Two random choices** instead of the global best, so routers rarely agree.
- **Optimistic local accounting:** when a router sends a request, it immediately adds the request's tokens to its own estimate of that replica's load, without waiting for the next scrape.
- **Jitter** on scrape times.

## Benefits

Queue-aware routing cuts TTFT tails noticeably under bursty load, because requests avoid replicas with long queues or nearly full KV. It does not change average throughput much; it changes who waits.

## Deep dive: herding with stale load reports

Each replica reports its load every ⟦m1048⟧ seconds. In between, a least-loaded router sends every new request to the same replica. The number sent before the next update is

⟦M1047⟧

At 200 req/s and 1 s reports, about 200 requests go to one replica, which then looks the busiest and gets nothing for the next second. Load oscillates instead of balancing.

~~~dg-raw
%%chart Requests in flight on two replicas under least-loaded routing with 1 s stale reports (illustrative).
x = 0, 1, 2, 3, 4, 5, 6
xlab = Seconds
ylab = Requests in flight
Replica A: 40, 240, 90, 250, 80, 245, 85
Replica B: 40, 50, 240, 70, 250, 75, 250
%%
~~~

Fixes: pick the best of two random replicas (each router choice is randomized, so a stale view cannot send everyone to one place), count requests the router itself has sent since the last report, or both.

<div class="take">
<div class="lab">Key takeaway</div>

route on queue and KV signals, but expect the metrics to be stale. Two random choices and local accounting avoid herding.

</div>
