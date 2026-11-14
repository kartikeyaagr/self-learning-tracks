---
id: "18.05"
course: "inference-eng-course"
module: "18"
slug: "run-and-benchmark-two-gpu-inference"
title: "Run and Benchmark Two-GPU Inference"
keyIdeas: ["Run the same model at TP 1 (one GPU, and two replicas) and TP 2.", "Compare TPOT at low load, max SLO-safe throughput, and throughput per GPU.", "Check topology and NCCL logs first.", "TP buys latency and memory; replicas buy throughput.", "Profile NCCL time per step at TP 2."]
question: "You run an 8B model with TP 1 on one GPU and TP 2 on two GPUs. TPOT improves 1.5 times but throughput per GPU drops. Is that a bad result?"
answer: "Not necessarily. TP 2 cuts per-step weight reads in half, so latency improves, but communication overhead means two GPUs give less than double the throughput of one. If your SLO needs the lower TPOT, TP 2 is right; if you only need throughput and TP 1 meets the SLO, two TP-1 replicas give more total throughput per GPU."
sources: [{"title": "vLLM: Parallelism and scaling", "url": "https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html", "note": "Running with tensor-parallel-size."}, {"title": "nccl-tests", "url": "https://github.com/NVIDIA/nccl-tests", "note": "Baseline communication speed."}]
tags: ["tensor-parallel", "lab"]
---

## Goal

Measure, for a model that fits on one GPU, what tensor parallelism of 2 buys and costs compared with running two independent replicas on the same two GPUs. This is the most common real decision about TP, and the answer depends on the SLO.

## Setup

- A model that fits on one GPU with room for KV cache, such as an 8B model in BF16.
- Two GPUs connected by NVLink if possible. If you only have PCIe, run the lab anyway and note the difference.
- The same engine version, workload and SLO for every configuration.

## The three configurations

- **A**: one replica with TP 1 on one GPU. The baseline.
- **B**: two TP 1 replicas, one per GPU, behind a simple least-loaded load balancer.
- **C**: one replica with TP 2 across both GPUs.

```
vllm serve <model> --tensor-parallel-size 2
```

## Measurements

| Metric | A | B | C |
| --- | --- | --- | --- |
| TPOT at concurrency 1 | | | |
| p95 TPOT at the expected load | | | |
| SLO-safe requests per second | | | |
| Requests per second per GPU | | | |
| KV capacity per replica (tokens) | | | |
| Communication share of step time | 0 | 0 | |

Get the KV capacity from each startup log, and the communication share from one Nsight Systems capture of C at moderate load.

## Expected pattern

- **C has the lowest TPOT at low load**, because each GPU reads half the weights.
- **C has the most KV capacity per replica**, because each GPU stores half the weights and half the KV per token, so more tokens fit per replica.
- **B has the most total throughput** at a relaxed SLO, because it has no communication and two independent schedulers.
- **C's throughput per GPU is lower than A's**, because of the all-reduces and smaller matmuls.

<div class="ex">
<div class="lab">Worked example</div>

**one set of results**

8B model, two H100s with NVLink, chat workload, SLO p95 TPOT under 30 ms.

A: TPOT at concurrency 1 is 6.1 ms, SLO-safe capacity 11 req/s.

B: same TPOT, 21.5 req/s in total (10.75 per GPU).

C: TPOT at concurrency 1 is 3.9 ms, SLO-safe capacity 17.5 req/s (8.75 per GPU). All-reduces were 14% of step time.

With a stricter SLO of p95 TPOT under 8 ms: A and B manage only about 1.5 req/s per GPU before TPOT exceeds 8 ms, while C manages about 6 req/s in total. Under that SLO, C wins clearly.

</div>

## Deciding

- Choose **C** when the latency SLO needs it, or when long contexts need its larger KV capacity per replica.
- Choose **B** otherwise. More replicas usually mean more throughput per GPU.

Repeat the decision whenever the SLO or the typical context length changes.

## Deep dive: scaling efficiency

Report every multi-GPU result against the ideal:

⟦M913⟧

A TP 8 setup with 5 times the throughput of one GPU is 62% efficient. That can still be the right choice when the model does not fit on fewer GPUs or the latency target demands it, but the 38% is the price, and it belongs in the write-up.

<div class="take">
<div class="lab">Key takeaway</div>

for a model that fits on one GPU, tensor parallelism trades throughput per GPU for lower latency and more KV capacity per replica. Measure both layouts under your SLO and let the SLO decide.

</div>
