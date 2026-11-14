---
id: "22.03"
course: "inference-eng-course"
module: "22"
slug: "warm-up-compilation-readiness"
title: "Warm-Up, Compilation & Readiness"
keyIdeas: ["Loading weights is not the same as ready to serve fast.", "Warm up with representative requests: short and long prompts, typical batch sizes.", "Report ready only after warm-up completes.", "Use a startup probe with a long timeout so slow starts are not killed.", "Liveness should check the process, not latency, to avoid restart storms."]
question: "Why should readiness be reported only after warm-up requests, not after the model loads?"
answer: "The first requests after load trigger lazy compilation, kernel autotuning, allocator growth and cache population, so they can be many times slower. If the load balancer sends real traffic as soon as weights load, those users get very slow responses or timeouts. Running synthetic warm-up requests first and then reporting ready keeps real users on warmed replicas."
sources: [{"title": "Kubernetes: Liveness, readiness and startup probes", "url": "https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/", "note": "Probe types."}, {"title": "vLLM: Optimization and tuning", "url": "https://docs.vllm.ai/en/latest/configuration/optimization.html", "note": "Compile and graph settings."}]
tags: ["readiness"]
---

## Three states

| State | Meaning | Probe |
| --- | --- | --- |
| Starting | loading weights, compiling | startup probe with a long timeout |
| Ready | warmed up and accepting traffic | readiness probe |
| Alive | process healthy | liveness probe |

## Why "loaded" is not "ready"

Right after loading, the first requests may trigger:

- Kernel autotuning for shapes not yet seen.
- Lazy compilation or graph capture for new batch sizes.
- First-time allocation of memory pools.
- Loading of tokenizer files, grammars or LoRA adapters.

Each of these can add hundreds of milliseconds to several seconds. If readiness flips as soon as weights are loaded, real users pay for them.

## Warm-up script

1. Send a short request and a long request, close to your longest real prompt.
2. Send a burst of concurrent requests that covers typical batch sizes.
3. Exercise every feature you use: structured output, LoRA adapters, multimodal input, tool calls.
4. Check that the responses are correct, not only that they returned.
5. Then flip readiness to true.

<div class="ex">
<div class="lab">Worked example</div>

**the cost of skipping warm-up**

A replica reports ready right after loading. The first JSON-schema request compiles a grammar (about 800 ms), and the first batch of 48 captures a new CUDA graph (about 1.5 s).

For the first minute, about 1 in 20 users sees TTFT 2 to 3 seconds higher than normal.

A 20-second warm-up script hitting these paths removes the spike.

</div>

## Pitfalls

- **Liveness based on latency:** under heavy load, responses slow down, the liveness probe fails, the pod restarts, and the remaining pods get even more load. Keep liveness a simple "is the process responding" check.
- **Readiness too early:** first users pay for compilation.
- **Short startup timeouts:** large models get killed before they finish loading, then restart and get killed again.

## Deep dive: what skipping warm-up costs the first users

The first requests on a new replica pay for one-time work: kernel autotuning, allocator growth, lazy CUDA module loading, and graph capture for batch sizes not yet seen. If ⟦m1014⟧ requests are affected and each is slower by ⟦m1015⟧:

⟦M1013⟧

With 20 affected requests per start, 50 starts a day and 2 s extra each, 1,000 requests a day see a 2 s stall. At 1 million requests a day that is only 0.1%, but it lands entirely in p99.9, and it lands on whoever was routed to the new replica first. A warm-up script that sends those 20 requests itself moves the cost off users.

<div class="take">
<div class="lab">Key takeaway</div>

ready means warmed up, not just loaded. Warm up with realistic requests before taking traffic, and keep liveness checks simple.

</div>
