---
id: "06.03"
course: "ml-system-design"
module: "06"
slug: "latency-throughput-and-cost"
title: "Latency, throughput and cost"
keyIdeas: ["Latency is set at a percentile (usually p99); throughput is requests or items per second per machine; cost is what that capacity costs. Optimising one often hurts another.", "Batching raises throughput, especially on GPUs, but adds waiting time. Choose the maximum batch size and wait time from the latency budget.", "The usual levers are smaller or compressed models (distillation, quantisation, pruning), caching, precomputation, cascades, feature-fetch optimisation and hardware choice.", "Measure where time goes before optimising. Feature fetching and network often cost more than the model."]
question: "A text classifier on GPU serves 2,000 requests per second with p99 latency of 180 ms; the budget is 100 ms, and the GPU bill is too high. Profiling shows 60 ms in tokenisation and feature fetch on CPU, 15 ms model time per batch, and the rest queueing. What would you change, in what order?"
answer: "First reduce queueing, which is most of the tail: check the dynamic batcher's maximum wait and batch size; with 2,000 RPS a 5 ms wait already gives batches of about 10, so a long wait or an oversized max batch only adds delay. Add capacity or autoscaling if utilisation is near saturation, since queueing grows sharply near 100%. Second, attack the 60 ms of CPU work: parallelise or move tokenisation to a fast library, cache features for repeated inputs, fetch features in one batched call, and run these concurrently with other work. Third, make the model cheaper: distil to a smaller model, quantise to int8 or fp16 with a quality check, and use a shorter maximum sequence length if inputs allow. This cuts GPU time per request so fewer GPUs are needed, lowering cost. Re-measure p99 after each change; also consider whether a cascade (a tiny model handles easy requests, the large one only uncertain ones) can cut GPU load further."
sources: [{"title": "Google SRE Book, Service Level Objectives", "url": "https://sre.google/sre-book/service-level-objectives/", "note": "Choosing latency targets and percentiles."}, {"title": "Dean and Barroso, The Tail at Scale (CACM 2013)", "url": "https://research.google/pubs/the-tail-at-scale/", "note": "Why fan-out makes tail latency worse, and how to reduce it."}, {"title": "NVIDIA Triton Inference Server: dynamic batching", "url": "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html", "note": "Configuring batch size and queue delay."}]
practice: ["p50-p95-and-p99-inference-latency", "dynamic-request-batching", "int8-symmetric-quantize", "optimize-throughput-and-cost-under-latency-slo", "estimate-model-memory-and-minimum-gpu-count"]
tags: ["latency", "throughput", "cost optimisation"]
---

## Three quantities that pull against each other

- **Latency**: how long one prediction takes, reported at p50, p95 and p99. Users feel the tail.
- **Throughput**: predictions per second per machine or GPU.
- **Cost**: compute, memory and network per prediction or per month.

A bigger batch raises throughput and lowers cost per prediction but makes each request wait. A bigger model raises quality and cost. Caching lowers both latency and cost but serves stale results. Design means picking a point that meets the service level objective (SLO) at acceptable cost.

## Where the time goes

Profile the whole request, not just the model:

| Stage | Typical share | Levers
| --- | --- | --- |
| Network and serialisation | 5-20% | co-locate services, compact formats
| Feature fetching | 20-50% | batch lookups, cache hot keys, precompute
| Preprocessing (tokenising, resizing) | 5-30% | fast libraries, move to accelerator
| Queueing | 0-60% | capacity, autoscaling, batch-wait tuning
| Model inference | 10-50% | smaller model, quantisation, batching
| Post-processing and rules | 1-10% | simplify

Often the model is not the bottleneck. A 2 ms model behind three sequential 20 ms feature calls is a feature problem.

## Queueing and utilisation

As utilisation approaches 100%, queueing delay grows without bound. Running servers at 60-70% of capacity at peak keeps the tail under control. Little's law, $L = \lambda W$, links requests in flight $L$, arrival rate $\lambda$ and time in system $W$, and is useful to size concurrency limits.

## Batching

GPUs process a batch of 32 inputs almost as fast as 1, so batching multiplies throughput. Dynamic batching waits up to a maximum time to fill a batch:

~~~dg-flow Dynamic batching. The maximum wait sets the extra latency; the batch size sets throughput.
Requests -(queue)-> Batcher -(full batch or max wait reached)-> *Model on GPU -(split results)-> Responses
~~~

At 2,000 requests per second, a 5 ms wait collects about 10 requests. Choose the maximum wait so that wait plus model time fits within the p99 budget.

## Making the model cheaper

| Technique | Typical gain | Risk
| --- | --- | --- |
| Distillation to a smaller model | 2-10× faster | some quality loss; needs training
| Quantisation (int8, fp16, 4-bit) | 1.5-4× faster, less memory | accuracy loss, especially at 4-bit
| Pruning and sparsity | varies with hardware support | needs fine-tuning
| Shorter inputs (sequence length, image size) | roughly linear or quadratic savings | lose information
| Caching outputs for repeated inputs | large for skewed traffic | staleness, memory
| Cascades | heavy model only on hard cases | routing errors

Always re-check quality after each change on the same evaluation set and slices.

## Tail latency at scale

When a request fans out to many backends, the slowest dominates. If each of 100 shards has a 1% chance of being slow, about 63% of requests hit at least one slow shard. Mitigations: hedged requests (send a duplicate after a short delay and take the first response), timeouts with partial results, and reducing fan-out.

<div class="ex">
<div class="lab">Worked example</div>

**cost cut by caching**

A product-description classifier ran on every page view, though descriptions changed rarely. Caching predictions by a hash of the description, with invalidation on edit, served 97% of requests from cache. GPU use fell by a factor of 20 and p99 latency dropped from 45 ms to 4 ms for cached items.

</div>

## Hardware choices

CPUs are cheap and flexible for small models and trees. GPUs win for large neural models with batching. For very high volume, specialised accelerators or optimised runtimes (ONNX Runtime, TensorRT) can cut cost further. Choose by cost per prediction at the required latency, measured on your model, not by peak FLOPs.

## Autoscaling

Traffic changes through the day, so fixed capacity sized for peak wastes money at night. Scale replicas on a signal that tracks load, such as requests per second or queue depth, rather than CPU alone, and keep a warm minimum so a sudden spike does not wait for new machines to load the model. For GPU models, start-up can take minutes, which makes scaling on predicted traffic worthwhile.

<div class="take">
<div class="lab">Key takeaway</div>

set a p99 SLO, profile the whole request before touching the model, keep utilisation below saturation, tune batching to the budget, and cut cost with smaller or quantised models, caching and cascades, checking quality after each change.

</div>
