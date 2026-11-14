---
id: "18.04"
course: "inference-eng-course"
module: "18"
slug: "collectives-communication-overhead"
title: "Collectives & Communication Overhead"
keyIdeas: ["Collectives: all-reduce (TP), all-gather, reduce-scatter, all-to-all (MoE), send/recv (PP).", "Cost = latency per call + bytes / bandwidth.", "In decode, messages are small, so latency dominates.", "Overlap communication with compute where possible; use custom all-reduce kernels for small messages.", "Measure with nccl-tests and in profiles (NCCL kernels in Nsight Systems)."]
question: "Estimate the all-reduce cost per decode step for a 80-layer model with hidden size 8192 at batch 32 with TP 4 over NVLink at 400 GB/s effective bus bandwidth."
answer: "Each all-reduce moves 32 x 8192 x 2 bytes = 512 KB. A ring all-reduce sends about 2 x (TP-1)/TP of the data per GPU, about 768 KB, taking roughly 2 microseconds at 400 GB/s plus around 10 to 20 microseconds of latency. Two per layer times 80 layers is 160 all-reduces, roughly 2 to 3.5 ms per step, mostly latency. That is significant against a 15 to 25 ms step."
sources: [{"title": "NCCL User Guide", "url": "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html", "note": "Collective algorithms."}, {"title": "nccl-tests", "url": "https://github.com/NVIDIA/nccl-tests", "note": "Measure collective bandwidth and latency."}]
tags: ["nccl", "communication"]
---

## The collectives

Multi-GPU inference uses a handful of collective operations. Each parallelism style relies on different ones.

| Collective | What it does | Used by |
| --- | --- | --- |
| All-reduce | sums a tensor across GPUs, and every GPU gets the result | tensor parallelism |
| All-gather | every GPU gets every other GPU's piece | sequence and context parallelism, some TP variants |
| Reduce-scatter | sums across GPUs, and each GPU keeps one slice | TP variants that overlap communication |
| All-to-all | every GPU sends different data to every other GPU | expert parallelism |
| Send and receive | point-to-point transfer | pipeline parallelism, KV transfer in disaggregation |

## A cost model

A simple model predicts collective time well enough to make decisions:

time ≈ α + bytes per GPU ÷ bandwidth

- **α** is the fixed latency per call: launching the kernel, synchronizing GPUs and the link latency. Typically 5 to 20 µs for NCCL.
- For a **ring all-reduce** over P GPUs, each GPU sends and receives about 2 × (P − 1) ÷ P times the message size.

For small messages, α dominates. For large messages, bandwidth dominates.

## Decode is latency bound

Decode messages are small. At batch 32 with hidden size 8,192 in BF16, each all-reduce moves 32 × 8,192 × 2 bytes = 512 KB. Over NVLink at an effective 300 GB/s, the bandwidth term is under 3 µs, so α is most of the cost.

With 160 all-reduces per step for an 80-layer model, 10 to 15 µs each adds 1.6 to 2.4 ms per step. For a 10 ms step, that is a large share.

Engines reduce this with **custom all-reduce kernels** for small messages over NVLink. These kernels write directly into the other GPUs' memory and synchronize with flags, avoiding much of NCCL's general-purpose overhead. They can bring the latency of small all-reduces down to a few microseconds.

~~~dg-raw
%%widget tp Set the tokens per step low to see decode, then high to see prefill. Communication's share shifts as the message size changes.
model = Llama 3 70B
b = 3
%%
~~~

## Prefill is bandwidth bound

Prefill messages are large. A 4,096-token chunk with hidden size 8,192 gives 64 MB per all-reduce. At an effective 300 GB/s, that is about 0.4 ms, and α hardly matters. Across 160 all-reduces, prefill communication can add 60 ms or more to a chunk.

The standard mitigation is **overlap**: split the chunk into pieces and communicate one piece while computing the next, so communication hides behind compute.

<div class="ex">
<div class="lab">Worked example</div>

**counting communication in a profile**

Llama 3 70B at TP 8 on an H100 server, decode at batch 64. Step time: 11 ms.

Nsight Systems shows 160 all-reduce kernels per step averaging 14 µs: 2.2 ms, 20% of the step.

Switching on the engine's custom all-reduce path cut them to 6 µs each: 1.0 ms, saving 1.2 ms per step, an 11% TPOT improvement.

</div>

## Measuring

- **nccl-tests** (`all_reduce_perf`, `alltoall_perf`) across message sizes gives the raw curve for your hardware. Note where the latency floor ends and bandwidth takes over.
- **Nsight Systems** shows NCCL or custom all-reduce kernels in each step. Sum their time and divide by the step time.
- **Count calls**: multiply the number of collectives per layer by the number of layers. That count, times α, is the minimum communication cost per step.

## Deep dive: communication in a prefill step

In prefill, each all-reduce carries the full activation, ⟦m909⟧ bytes. For Llama 3 70B (⟦m910⟧) and ⟦m911⟧:

⟦M908⟧

Times 160 all-reduces: 83 ms. The prefill compute on 8 GPUs is ⟦m912⟧ ms. Communication is about a quarter of the step unless it is overlapped with compute, which is why engines split the matmuls and start sending the first part of the result while computing the rest.

<div class="ex">
<div class="lab">Worked example</div>

**counting all-reduces in a profile**

An Nsight Systems trace of one decode step for a 70B model at TP 8 should show 160 NCCL all-reduce kernels, two per layer, plus one or two for the logits.

If it shows 320, something (often a norm or the residual add) is all-reducing again. If it shows each all-reduce at 60 µs instead of 15 µs, the GPUs are communicating over PCIe, not NVLink: check the topology.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

communication costs a fixed latency per call plus bytes over bandwidth. Decode pays mostly latency, so count calls per step and use low-latency all-reduce kernels. Prefill pays mostly bandwidth, so overlap communication with compute.

</div>
