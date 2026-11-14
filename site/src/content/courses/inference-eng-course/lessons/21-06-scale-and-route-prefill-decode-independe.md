---
id: "21.06"
course: "inference-eng-course"
module: "21"
slug: "scale-and-route-prefill-decode-independently"
title: "Scale and Route Prefill/Decode Independently"
keyIdeas: ["Each pool scales on its own signals.", "Prefill: prompt tokens queued, TTFT; decode: KV usage, concurrency, TPOT.", "Routing picks prefill workers by load and cache hits, decode workers by KV headroom.", "Some systems let workers switch roles to follow mix shifts.", "Keep minimum capacity in both pools for bursts."]
question: "Which signals would you use to autoscale the prefill pool and the decode pool separately?"
answer: "Prefill pool: queued prompt tokens or prefill queue wait time, and TTFT percentile against SLO. Decode pool: KV cache usage, running sequences per worker, and TPOT percentile against SLO. Scale each when its signals approach thresholds; request rate alone is insufficient because the prompt/output mix changes."
sources: [{"title": "NVIDIA Dynamo", "url": "https://github.com/ai-dynamo/dynamo", "note": "Planner for disaggregated scaling."}, {"title": "llm-d", "url": "https://github.com/llm-d/llm-d", "note": "Kubernetes-native distributed inference with disaggregation."}]
tags: ["disaggregation", "autoscaling"]
---

## Signals per pool

| Pool | Scale up when | Scale down when |
| --- | --- | --- |
| Prefill | prompt tokens waiting grow, TTFT p95 near the SLO | prefill GPUs are idle much of the time |
| Decode | KV usage above about 85%, TPOT p95 near the SLO | low KV usage and low concurrency |

GPU utilization alone is a poor signal for either pool. A decode GPU can show 100% utilization while having plenty of KV room, and a prefill GPU can look busy while its queue is empty.

## Routing

- **Prefill choice:** the fewest queued prompt tokens, or the best prefix cache hit.
- **Decode choice:** the most free KV blocks and a running batch below its limit.
- **Pairing:** prefer a prefill-decode pair with a fast link between them, such as the same node or rack.

~~~dg-flow Routing one request.
New request -> Prefill: best cache hit or shortest queue -> *Decode: most free KV -> Check the link between them
~~~

## Role switching

When the mix shifts, for example during a surge of RAG traffic, converting a decode worker into a prefill worker avoids a slow cold start of a new GPU.

- Workers must be able to run either role.
- A worker must drain cleanly: finish or migrate its running sequences before switching.
- Switching too often wastes time on draining, so add hysteresis: switch only after a signal stays high for a few minutes.

## Deep dive: sizing each pool from its own queue

Apply Little's law to each pool separately:

⟦M1004⟧

⟦m1005⟧ is time in the prefill pool (queue plus prefill), ⟦m1006⟧ is the decode duration of a request (⟦m1007⟧ TPOT). At 30 req/s and 300 output tokens at 15 ms, ⟦m1008⟧ s, so 135 sequences decode at once; with ⟦m1009⟧ per GPU the decode pool needs at least 3 GPUs. Scale the prefill pool on its queue length and TTFT, and the decode pool on its running count and KV usage. They respond to different signals.

<div class="take">
<div class="lab">Key takeaway</div>

disaggregated pools scale and route on phase-specific signals. Queued prompt tokens and TTFT drive prefill; KV usage and TPOT drive decode.

</div>
