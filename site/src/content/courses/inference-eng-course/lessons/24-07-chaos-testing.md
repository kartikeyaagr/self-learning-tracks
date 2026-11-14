---
id: "24.07"
course: "inference-eng-course"
module: "24"
slug: "chaos-testing"
title: "Chaos Testing"
keyIdeas: ["Chaos testing injects failures deliberately to verify recovery.", "Start with a hypothesis: \"if X fails, the SLO holds within Y.\"", "Run in staging first, then carefully in production with small blast radius.", "LLM-specific chaos: long-prompt bursts, KV exhaustion, slow replicas, cache tier loss.", "Each experiment should end in a fix or a confirmed behavior."]
question: "Name three chaos experiments worth running on an LLM fleet and the expected safe outcome for each."
answer: "Kill one replica under load: router stops sending within seconds, SLO dips briefly, in-flight requests retry successfully. Inject a slow replica (throttle its GPU or add latency): queue-aware routing shifts traffic away and p99 stays bounded. Send a burst of very long prompts: admission control and chunked prefill keep other users within SLO and no OOM occurs."
sources: [{"title": "Principles of Chaos Engineering", "url": "https://principlesofchaos.org/", "note": "The method."}, {"title": "Chaos Mesh", "url": "https://chaos-mesh.org/", "note": "Kubernetes fault injection."}]
tags: ["reliability", "chaos"]
---

## Method

1. **Define steady state:** SLO attainment and error rate under normal load.
2. **State a hypothesis:** steady state holds when this fault happens.
3. **Inject the fault** with a limited blast radius, for example one replica out of ten.
4. **Observe,** and abort if impact goes beyond agreed bounds.
5. **Fix weaknesses** and repeat.

## LLM-specific experiments

| Fault | Hypothesis |
| --- | --- |
| Kill a replica | traffic reroutes within 10 s; errors are retried |
| Slow replica | routing avoids it |
| Burst of very long prompts | no OOM; other users' ITL p99 stays bounded |
| Fill KV with long outputs | preemption or shedding, no crash |
| Cache tier down | TTFT rises, no errors |
| Network partition in a TP group | the replica restarts cleanly |

<div class="ex">
<div class="lab">Worked example</div>

**a long-prompt burst**

Steady state: 50 requests/s, ITL p99 of 40 ms, no errors.

Fault: 20 requests with 100k-token prompts arrive within 10 seconds.

A typical result: two replicas hit CUDA OOM because the maximum batched tokens setting allowed a single 100k prefill plus a full decode batch. ITL p99 on the others rose to 300 ms because of long prefill chunks.

Fixes: cap maximum batched tokens, enable chunked prefill with a smaller chunk, and route very long prompts to a dedicated pool. Re-run the test and confirm.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

chaos experiments turn assumed resilience into tested resilience. Include LLM-specific faults such as long-prompt bursts and KV exhaustion.

</div>
