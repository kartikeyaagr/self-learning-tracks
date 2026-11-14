---
id: "18.09"
course: "inference-eng-course"
module: "18"
slug: "lab-choose-and-benchmark-a-parallelism-strategy"
title: "Lab: Choose and Benchmark a Parallelism Strategy"
keyIdeas: ["List configurations that fit: TP x replicas (x PP across nodes).", "For each, measure node-level goodput under the SLO and TPOT at low load.", "Check memory: weights per GPU and KV capacity per replica.", "Profile communication share per step.", "Choose by goodput per node within the SLO."]
question: "For a 70B model on one 8-GPU H100 node serving chat traffic, which configurations would you test and how do you choose?"
answer: "TP 8 as one replica, TP 4 x 2 replicas, and TP 2 x 4 replicas if weights in FP8 fit (70 GB over 2 GPUs leaves some KV room). Measure SLO-safe goodput for the node, TPOT at low load, and KV capacity per replica. Choose the configuration with the highest node goodput that meets the TPOT SLO."
sources: [{"title": "vLLM: Parallelism and scaling", "url": "https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html", "note": "Configuring TP, PP and DP."}, {"title": "nccl-tests", "url": "https://github.com/NVIDIA/nccl-tests", "note": "Communication baselines."}]
tags: ["lab"]
tools: ["Lab"]
---

## Goal

Choose the best parallelism layout for a large model on one 8-GPU server, using measurements. The model is Llama 3 70B (or similar), and the server has eight 80 GB GPUs connected by NVLink.

## Candidate configurations

| Config | Replicas | Weights per GPU | KV capacity per replica |
| --- | --- | --- | --- |
| TP 8, BF16 | 1 | 17.5 GB | large |
| TP 4, BF16 | 2 | 35 GB | medium |
| TP 2, FP8 weights | 4 | 35 GB | smaller |

Every configuration uses all 8 GPUs. TP 2 in BF16 does not fit (70 GB of weights per GPU leaves almost no room for KV), so it uses FP8 weights, and its quality must be checked.

## Measurements

For each configuration:

- **Server goodput at the SLO**: the sum across all replicas, with a least-loaded router in front.
- **TPOT p50 at concurrency 1**: the best latency each layout offers.
- **Maximum concurrency per replica** before KV pressure causes preemption, at your typical context length.
- **Communication share of step time**, from an Nsight Systems capture at moderate load.
- **Quality**, for the FP8 configuration, against the BF16 baseline.

## Typical outcome

| Config | TPOT at low load | Server goodput |
| --- | --- | --- |
| TP 8 | best | lowest |
| TP 4 × 2 | good | higher |
| TP 2 × 4 (FP8) | acceptable | often highest, if quality allows |

Why: higher TP lowers each step's weight read per GPU, which helps latency, but adds communication and reduces the number of independent schedulers. More replicas use the GPUs more efficiently for throughput.

<div class="ex">
<div class="lab">Worked example</div>

**one set of results**

Chat workload, SLO p95 TPOT under 40 ms.

TP 8: TPOT at concurrency 1 is 8.5 ms. Communication 27% of step. Server goodput 5,400 tok/s.

TP 4 × 2: TPOT 14 ms. Communication 15%. Server goodput 7,900 tok/s.

TP 2 × 4 with FP8: TPOT 13 ms (FP8 halves the weight read). Communication 8%. Server goodput 10,600 tok/s. Product eval within 0.4 points of BF16.

With the 40 ms SLO, TP 2 × 4 serves about twice as much as TP 8. With a 12 ms TPOT SLO, only TP 8 meets it at all.

</div>

## Write-up

- Recommend a configuration, with the goodput and TPOT charts and the quality result.
- State when you would switch: a stricter TPOT SLO pushes toward higher TP, longer contexts may push toward fewer, larger replicas for KV capacity, and a quality regression rules out FP8.
- Include the communication share for each layout as the explanation for the goodput differences.

## Deep dive: cost per token across configurations

Compare configurations on the cost of a million tokens at the SLO, not on raw throughput:

⟦M938⟧

| Configuration (70B) | GPUs | SLO-safe tok/s | Cost per M tokens at \$2.50 |
| --- | --- | --- | --- |
| TP 8 × 1 replica | 8 | 9,000 | \$0.62 |
| TP 4 × 2 replicas | 8 | 11,500 | \$0.48 |
| TP 2 FP8 × 4 replicas | 8 | 13,000 | \$0.43 |

Illustrative numbers. The pattern is common: the smallest TP that fits the model and meets the TPOT target usually gives the lowest cost, because communication overhead is paid per GPU.

<div class="take">
<div class="lab">Key takeaway</div>

for a fixed server, more replicas usually mean more throughput and higher TP means lower latency. Measure server goodput under the SLO, and pick the most replicas the SLO and quality allow.

</div>
