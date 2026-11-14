---
id: "IN.02"
course: "inference-eng-course"
module: "IN"
slug: "the-inference-stack-model-to-gpu-to-production"
title: "The Inference Stack: Model to GPU to Production"
keyIdeas: ["Think of inference as a stack: API, routing, engine scheduler, KV memory, model runtime, kernels, hardware.", "Every layer has a characteristic optimization and a characteristic metric.", "Bottlenecks move between layers. Fixing the kernel layer does nothing if the scheduler leaves the GPU idle.", "Production adds a fleet layer on top: replicas, autoscaling, rollouts and observability.", "Knowing which layer a symptom comes from is most of debugging."]
question: "List the layers a single chat request passes through from the user's HTTP call to GPU kernels, and name one optimization that lives at each layer."
answer: "Client and API gateway (rate limiting, auth), router or load balancer (cache-aware routing), inference engine scheduler (continuous batching), KV cache manager (paged allocation, prefix caching), model runtime (CUDA graphs, torch.compile), kernels (FlashAttention, quantized GEMM), and hardware (GPU memory and NVLink). Each layer has its own optimizations and its own metrics."
sources: [{"title": "vLLM documentation", "url": "https://docs.vllm.ai/en/latest/", "note": "A full engine you can map onto this stack."}, {"title": "Modal GPU Glossary", "url": "https://modal.com/gpu-glossary", "note": "Short definitions of GPU hardware and software terms."}]
tags: ["stack"]
---

## The layers of a request

A chat request touches many layers before a token comes back. Naming them lets you say where a problem lives, which is most of debugging.

~~~dg-layers The inference stack, top to bottom.
API and gateway | auth, rate limits, OpenAI-compatible endpoints, streaming
Router and fleet | load balancing, cache-aware routing, autoscaling
*Engine scheduler | which requests run this step, batching, preemption
KV cache manager | paged blocks, prefix reuse, offload
Model runtime | forward pass, CUDA graphs, compilation
Kernels | attention, GEMM, fused norms, quantized matmuls
Hardware | HBM, SMs, tensor cores, NVLink, PCIe
~~~

## Following one request down the stack

Here is what happens when a user sends "Summarize this contract" with a 6,000-token contract attached. The gateway has already checked auth and rate limits; the diagram starts at the router.

~~~dg-seq One request through the stack. Press Play to step through it.
Client > Router : POST /chat
Router > Engine : replica with cache
Engine > Engine : allocate KV blocks
Engine > GPU : prefill 6k tokens
GPU -> Engine : last logits
Engine -> Client : first token
Engine > GPU : decode step
GPU -> Engine : 1 token each
Engine -> Client : stream token
Engine > Engine : free KV blocks
~~~

## What each layer controls, and what goes wrong there

- **API and gateway.** Request format, streaming protocol (usually server-sent events), auth, rate limits, timeouts. A bad timeout here can kill long generations that were fine.
- **Router.** Picks the replica. A router that knows which replica already holds a prompt prefix in cache can skip seconds of prefill. A round-robin router throws that away.
- **Engine scheduler.** Decides every step which sequences advance and how many new prompt tokens to admit. This is where continuous batching lives and where the latency versus throughput trade-off is actually set.
- **KV cache manager.** Stores the attention keys and values for every active token. Its capacity decides how many requests can run at once.
- **Model runtime.** Launches the model's kernels in order. At small batch sizes, Python and kernel launch overhead can take a third of each step; CUDA graphs remove most of it.
- **Kernels.** The math itself: matrix multiplies, attention, normalization. Good kernels move fewer bytes and keep tensor cores busy.
- **Hardware.** Sets the hard ceilings: memory capacity (80 GB on an H100), memory bandwidth (3.35 TB/s), compute (about 990 dense BF16 TFLOP/s) and interconnect speed (NVLink about 450 GB/s each way per GPU).

## Reading symptoms by layer

| Symptom | Most likely layer | First thing to check |
| --- | --- | --- |
| TTFT rises sharply under load, decode speed fine | Scheduler or queueing | Queue length, waiting requests |
| Out-of-memory only with long prompts | KV manager | KV blocks free, max model length |
| GPU utilization under 40% at batch 1 | Runtime overhead | Kernel launch gaps in a profile |
| Tokens slow at every batch size | Kernels or bandwidth | Achieved memory bandwidth |
| One replica hot, others idle | Router | Routing policy, sticky sessions |
| Errors spike during deploys | Fleet and lifecycle | Readiness probes, draining |

<div class="ex">
<div class="lab">Worked example</div>

**a TTFT spike that was not the GPU**

A team sees p95 TTFT jump from 400 ms to 4 s at peak. Kernels are unchanged and GPU utilization is 95%.

The scheduler metrics show 60 requests waiting while 128 are running. TTFT now includes 3.5 s of queueing.

The fix is in the router and autoscaler: add a replica at 40 queued requests. Making kernels 10% faster would not have touched the 3.5 s.

</div>

## Bottlenecks move

Fix one layer and the bottleneck moves to another. Speed up attention kernels and you may become limited by the scheduler admitting too few requests. Add replicas and you may become limited by router imbalance. Good inference work is a loop: measure end to end, find the layer that dominates, fix it, measure again.

## Deep dive: a latency budget, term by term

Every millisecond a user waits belongs to one layer of the stack. For a streamed response of ⟦m20⟧ tokens:

⟦M18⟧

⟦M19⟧

Each term has a different owner and a different fix. Writing the budget down before optimizing tells you which layer to work on.

<div class="ex">
<div class="lab">Worked example</div>

**budgeting a 1.5 s TTFT target**

A RAG request: 6,000 prompt tokens, 400 output tokens, 8B model on an H100.

Network and TLS: 40 ms. Gateway auth and routing: 5 ms. Tokenization: 3 ms.

Prefill: ⟦m21⟧ FLOPs. At 50% of 989 TFLOP/s, about 195 ms.

That leaves about 1,250 ms for queueing. If p95 queueing under load is 1.8 s, the fix is capacity or admission control, not a faster kernel.

E2E: with TPOT of 20 ms, the 400 tokens add ⟦m22⟧ s. For this request, decode is 97% of the total time, but TTFT is what the user notices first.

</div>

~~~dg-raw
%%bars The same request's latency split by layer (E2E, ms).
unit = ms
Network + gateway + tokenize | 48 | c4
Queue (p50 at moderate load) | 120 | flame
Prefill | 195 | c2
Decode (399 tokens × 20 ms) | 7980 | c1
%%
~~~

~~~dg-raw
%%timeline One request through the stack, first 600 ms.
unit = ms
Client | 0-20 send c4 ; 383-403 first token c4
Gateway | 20-25 auth c4 ; 25-28 tok c4
Scheduler | 28-148 waiting flame
GPU | 148-343 prefill c2 ; 343-363 step 1 c1 ; 363-383 step 2 c1 ; 383-403 step 3 c1 ; 403-423 step 4 c1 ; 423-443 step 5 c1
%%
~~~

## Where the variance lives

Fixed costs like tokenization barely vary. Queueing varies the most, because it depends on what other users are doing. That is why p99 TTFT is usually a capacity question and p50 TTFT is usually a prefill question.

<div class="take">
<div class="lab">Key takeaway</div>

name the layer before you optimize. Most wasted effort comes from tuning a layer that is not the bottleneck.

</div>
