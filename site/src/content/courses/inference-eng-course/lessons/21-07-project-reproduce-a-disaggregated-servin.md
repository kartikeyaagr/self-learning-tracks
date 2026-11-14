---
id: "21.07"
course: "inference-eng-course"
module: "21"
slug: "project-reproduce-a-disaggregated-serving-architecture"
title: "Project: Reproduce a Disaggregated Serving Architecture"
keyIdeas: ["Build or configure one prefill and one decode worker with a KV connector.", "Baseline: the same GPUs as two colocated replicas with chunked prefill.", "Workload: long prompts, moderate outputs, tight TPOT SLO.", "Measure goodput, TTFT, ITL p99 and transfer time.", "Report the crossover and the overheads."]
question: "What is the minimum setup to demonstrate disaggregation's benefit, and what chart proves it?"
answer: "At least two GPUs (one prefill, one decode) connected by a fast link, versus the same two GPUs as colocated replicas with chunked prefill, serving a workload with long prompts and a tight TPOT SLO. The proving chart is goodput (or SLO attainment) versus request rate for both setups, plus ITL p99 showing the interference removed."
sources: [{"title": "vLLM: Disaggregated prefilling", "url": "https://docs.vllm.ai/en/latest/features/disagg_prefill.html", "note": "A runnable starting point."}, {"title": "DistServe", "url": "https://arxiv.org/abs/2401.09670", "note": "Results to compare against qualitatively."}]
tags: ["project"]
tools: ["Project"]
---

## Goal

Build a small disaggregated system, compare it fairly with colocated serving on the same GPUs, and find the crossover.

## Setup

- 2 or more GPUs with NVLink or a fast network.
- **Disaggregated:** 1 prefill worker, 1 decode worker, a KV connector between them, and a simple router.
- **Colocated:** 2 replicas with chunked prefill, a tuned token budget and least-loaded routing.

## Workload

- Prompts of 4k to 8k tokens and outputs of 200 to 400 tokens.
- Poisson arrivals over a range of rates, from light load to past saturation.
- SLO: p95 TTFT under 2 s and p95 TPOT under 40 ms.

## Results to collect

| Metric | Colocated | Disaggregated |
| --- | --- | --- |
| Goodput at each rate | | |
| ITL p99 | | |
| TTFT p95 | | |
| KV transfer time p95 | not applicable | |

## What to expect

- ITL p99 should be clearly lower in the disaggregated setup at medium and high load, because no prefill ever stalls decode.
- TTFT may be slightly higher at low load in the disaggregated setup, because only one GPU does prefill.
- With only 2 GPUs, the 1:1 split may not match your traffic. Note how the result would change with a better ratio.

## Report

Include charts of goodput against rate and ITL p99 against rate. Discuss the crossover, the transfer overhead, and how the pool ratio should change for shorter or longer prompts.

## Deep dive: what the lab should show

| Measure | Colocated | Disaggregated | Why |
| --- | --- | --- | --- |
| p99 ITL under prompt bursts | Spikes with each burst | Flat | No prefill in decode steps |
| TTFT at low load | Lower | Slightly higher | Transfer and an extra hop |
| TTFT under prompt bursts | Rises sharply | Rises less | Prefill pool is not shared |
| Goodput at a tight TPOT SLO | Lower | Higher | 21.05 |
| GPUs idle at low load | Fewer | More | Two pools to keep warm |

<div class="take">
<div class="lab">Key takeaway</div>

reproducing disaggregation on two GPUs shows both its benefit (no interference) and its costs (transfer and pool sizing). The goodput crossover chart is the result.

</div>
