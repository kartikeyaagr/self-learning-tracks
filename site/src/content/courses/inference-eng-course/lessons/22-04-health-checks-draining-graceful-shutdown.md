---
id: "22.04"
course: "inference-eng-course"
module: "22"
slug: "health-checks-draining-graceful-shutdown"
title: "Health Checks, Draining & Graceful Shutdown"
keyIdeas: ["Health checks: alive vs ready; deep checks can run a tiny generation.", "Draining: stop new traffic, finish in-flight streams.", "Drain timeout should exceed typical long-request E2E time.", "Abort leftovers cleanly so clients can retry elsewhere.", "Long streams make draining slow; plan grace periods accordingly."]
question: "Describe a graceful shutdown sequence for an inference replica during a rolling deploy."
answer: "Mark the replica not ready so the load balancer stops sending new requests; keep serving in-flight requests until they finish or a drain timeout (longer than typical E2E latency) passes; abort remaining requests with a clear error so clients can retry; flush metrics and logs; then exit and free the GPU."
sources: [{"title": "Kubernetes: Pod lifecycle and termination", "url": "https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-termination", "note": "Termination grace period."}, {"title": "Google SRE Book: Handling Overload", "url": "https://sre.google/sre-book/handling-overload/", "note": "Graceful degradation."}]
tags: ["reliability"]
---

## Shutdown sequence

~~~dg-flow Graceful shutdown.
SIGTERM -> *Readiness false -> Finish in-flight requests (up to drain timeout) -> Abort the rest with a retryable error -> Exit
~~~

1. **SIGTERM arrives.** The platform is about to remove the pod.
2. **Readiness goes false.** The load balancer stops sending new requests. This takes a few seconds to propagate, so keep accepting stragglers briefly.
3. **In-flight requests finish,** up to the drain timeout.
4. **Remaining requests are aborted** with an error the client knows to retry.
5. **The process exits.**

## Drain timeout

LLM requests can stream for minutes. Set the grace period from your end-to-end latency distribution, for example p99 plus a margin.

<div class="ex">
<div class="lab">Worked example</div>

**sizing the grace period**

A chat service has p99 end-to-end latency of 45 s, and a few reasoning requests run 3 to 4 minutes.

A 60 s grace period completes 99% of streams. The long reasoning requests are cut off at every deploy.

Options: a 5-minute grace period (slow rollouts), a max-token cap that bounds the longest request, or routing reasoning traffic to a separate pool that is deployed less often.

</div>

## Health checks

| Check | What it does | How often |
| --- | --- | --- |
| Liveness | the process responds | frequently |
| Readiness | the engine accepts work and is not overloaded | frequently |
| Deep check | a tiny generation completes within a time budget | occasionally |

A deep check catches a wedged GPU or a hung engine loop that a plain HTTP check would miss. The HTTP server can respond happily while the engine loop has been stuck for minutes.

## The common failure

Pods killed before draining finishes cut users off mid-stream. Align the platform's grace period (for example terminationGracePeriodSeconds in Kubernetes) with the engine's drain logic, and test it under load.

## Deep dive: choosing the grace period

A request in flight when shutdown starts needs at most

⟦M1016⟧

to finish. If the grace period ⟦m1017⟧ is shorter, the share of in-flight requests that get cut is ⟦m1018⟧.

<div class="ex">
<div class="lab">Worked example</div>

**max-tokens 4,096 at 25 ms TPOT**

⟦m1019⟧ s. Kubernetes' default 30 s grace period cuts any request that still has more than about 1,150 tokens to generate.

Options: raise terminationGracePeriodSeconds to about 120 s, or cap max-tokens lower for the duration of a drain and let clients continue with a follow-up request.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

drain before you exit. Stop new traffic, finish streams, and abort the rest cleanly. Size the grace period from real end-to-end latency.

</div>
