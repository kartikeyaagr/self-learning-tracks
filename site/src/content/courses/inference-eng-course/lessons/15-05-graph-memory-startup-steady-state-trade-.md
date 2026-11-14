---
id: "15.05"
course: "inference-eng-course"
module: "15"
slug: "graph-memory-startup-steady-state-trade-offs"
title: "Graph Memory, Startup & Steady-State Trade-Offs"
keyIdeas: ["More captured sizes: less padding, more memory and longer startup.", "Aggressive autotuning: faster kernels, much longer compile.", "Graph memory pools reduce KV capacity slightly.", "Startup time matters for autoscaling and recovery.", "Cache compilation artifacts and choose capture sizes from real batch distributions."]
question: "You capture CUDA graphs for 67 batch sizes and compile with max-autotune. Startup now takes 6 minutes and KV capacity dropped 3%. When is that worth it?"
answer: "It is worth it for long-lived replicas with stable traffic, where steady-state speed matters over hours and startup is rare. It is not worth it when you autoscale often or need fast recovery, because every new replica takes 6 minutes before serving; there, capture fewer sizes, use default compile, or cache compiled artifacts."
sources: [{"title": "vLLM: Optimization and tuning", "url": "https://docs.vllm.ai/en/latest/configuration/optimization.html", "note": "CUDA graph and compile settings."}, {"title": "PyTorch: Compile time caching", "url": "https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html", "note": "Reusing compiled artifacts across runs."}]
tags: ["cuda-graphs", "startup"]
---

## The trade-offs

Compilation and CUDA graphs make steady-state serving faster, but they cost startup time and memory. How much each is worth depends on how your replicas live.

| Choice | Steady-state speed | Startup time | Memory |
| --- | --- | --- | --- |
| No compile, no graphs | slowest at small batch | fastest | most left for KV |
| Compile, few graph sizes | good | moderate | small cost |
| Compile, many graph sizes | best, least padding | long | larger cost |
| max-autotune compile | faster kernels | much longer | similar |

## How much startup costs

A rough breakdown of starting an 8B model replica:

- Container start and Python imports: 10 to 30 s.
- Loading 16 GB of weights from local NVMe: 5 to 15 s. From network storage: often minutes.
- Memory profiling and KV pool allocation: a few seconds.
- torch.compile: 20 s to a few minutes; much less with a warm cache.
- CUDA graph capture for 60 batch sizes: 10 to 60 s.
- Warm-up requests: a few seconds.

The total can range from under a minute to several minutes. Larger models with tensor parallelism take longer.

## Deciding by how replicas live

- **Long-lived, stable fleet.** Replicas run for days. Startup happens rarely, so optimize steady state: compile, capture many sizes, even autotune.
- **Spiky traffic with autoscaling.** New replicas start during traffic spikes. Each minute of startup is a minute of missing capacity when it is needed most, so keep startup short.
- **Frequent deploys.** Each deploy restarts every replica. A 5-minute startup across 100 replicas in rolling batches can stretch a deploy to hours.

<div class="ex">
<div class="lab">Worked example</div>

**when startup decides**

A service scales from 10 to 30 replicas when a traffic spike arrives. The spike reaches full height in 2 minutes.

Config A: startup 6 minutes, steady-state capacity 10 requests per second per replica. For 4 minutes after the spike, the service is short by up to 200 requests per second, and SLO attainment collapses.

Config B: startup 90 seconds, capacity 9 requests per second per replica, 10% slower. New replicas arrive in time; attainment stays above 99%.

B serves the spike better even though it is slower in steady state. The faster config would need more always-on replicas to cover the gap.

</div>

## Memory

CUDA graphs reserve memory for their intermediate tensors in a private pool. Many graph sizes, especially large ones, can take a few GB. That memory comes out of the KV pool, reducing concurrency. Check the KV capacity in the startup log with and without graphs.

## Reducing startup cost

- **Capture only the sizes you use.** If observed batches never exceed 96, capturing up to 512 wastes time and memory.
- **Cache compiled artifacts.** torch.compile and Triton cache their output on disk. With the same versions, model and shapes, a restart can reuse it. Bake the cache into the container image or a shared volume.
- **Load weights fast.** Keep weights on local NVMe or in the image; consider formats and loaders that read in parallel or stream directly to GPU memory.
- **Warm up before readiness.** Send representative requests before the replica reports ready, so the first users do not pay for lazy compilation (Module 23).

## Measure all of it

For each configuration, record:

- time from process start to ready,
- KV tokens available,
- TPOT at batch 1 and batch 64,
- SLO-safe capacity on your workload.

Then choose using your fleet's real behaviour: how often replicas start and how fast traffic changes.

## Deep dive: when compiling pays for itself

If compilation and capture add ⟦m819⟧ to startup and make the replica faster by a fraction ⟦m820⟧, the replica must live at least

⟦M818⟧

before the speed-up has repaid the time it spent starting. Below that, a faster start is worth more.

<div class="ex">
<div class="lab">Worked example</div>

**autoscaling against long-lived replicas**

Compile and capture add 90 s. They make decode 15% faster.

Break-even lifetime: ⟦m821⟧ s, ten minutes.

A replica that lives for hours should always compile. A burst replica that lives 5 minutes should not, unless the compiled artifacts are cached so the 90 s is paid once, not on every start.

</div>

~~~dg-raw
%%chart GPU-seconds gained or lost per replica start, by lifetime (90 s compile, 15% faster).
x = 120, 300, 600, 1800, 3600, 14400
logx = 1
xlab = Replica lifetime (s)
ylab = Net GPU-seconds saved
Net saving: -72, -45, 0, 180, 450, 2070
%%
~~~

<div class="take">
<div class="lab">Key takeaway</div>

compilation and CUDA graphs buy steady-state speed with startup time and memory. Choose based on how often replicas start and how quickly traffic changes, not on steady-state benchmarks alone.

</div>
