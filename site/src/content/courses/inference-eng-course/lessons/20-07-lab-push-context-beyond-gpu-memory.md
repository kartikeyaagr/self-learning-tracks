---
id: "20.07"
course: "inference-eng-course"
module: "20"
slug: "lab-push-context-beyond-gpu-memory"
title: "Lab: Push Context Beyond GPU Memory"
keyIdeas: ["Build a workload whose reusable KV exceeds GPU memory: long multi-turn chats or many repeated documents.", "Run GPU-only prefix caching, then add a CPU (and optionally SSD) tier.", "Measure TTFT, hit rate per tier, transfer bandwidth and goodput.", "Also run a no-reuse workload to check for overhead.", "Report when the tier helps and when it does not."]
question: "What result would show that your tiered KV setup is worth it?"
answer: "On a multi-turn or repeated-document workload whose total KV exceeds GPU memory, TTFT p95 and goodput improve compared with GPU-only caching, with hit rate in the CPU tier showing where the gain comes from, and with no regression on a workload with no reuse."
sources: [{"title": "LMCache", "url": "https://github.com/LMCache/LMCache", "note": "Offload layer to use in the lab."}, {"title": "vLLM: KV connector / offloading docs", "url": "https://docs.vllm.ai/en/latest/", "note": "Engine integration points."}]
tags: ["lab"]
tools: ["Lab"]
---

## Goal

Show, with numbers, when a KV tier beyond GPU memory pays off and what it costs when there is no reuse.

## Workload

- 200 conversations, each growing to 20k tokens or more over 10 turns.
- Turns arrive interleaved, so most conversations' KV is evicted from the GPU between turns.
- Back-of-envelope: 200 × 20k × 128 KB = 512 GB of reusable KV for an 8B model. That is ten times a single H100's KV pool, but fits in a server's CPU memory.

## Configurations

| Config | Description |
| --- | --- |
| A | prefix caching, GPU only |
| B | GPU plus a CPU tier through an offload connector |
| C | GPU plus CPU plus an SSD tier |

## Measure

- TTFT at p50 and p95 for each turn number.
- Hit rate in GPU, CPU and SSD tiers.
- Achieved transfer bandwidth and the number of requests waiting on loads.
- Goodput under your SLO.
- Overhead on a workload with no reuse.

## Expected results

- **Turn 1** is the same in every config.
- **Later turns:** A recomputes long histories, so TTFT grows with the history. B and C reload them, cutting TTFT several times for long histories.
- **No-reuse workload:** B and C should cost almost nothing. If they cost more than a few percent of throughput, the write-back path is blocking the GPU.

## Report

Plot TTFT p95 against turn number for each config, and give the per-tier hit rates. State the history length at which each tier starts to pay off.

## Deep dive: predicting the lab's outcome

With the formulas from this module, write the expected TTFT for a repeated 32k document before measuring:

| Configuration | Expected TTFT on a repeat | Source |
| --- | --- | --- |
| No prefix cache | About 1.3 s | Full prefill (20.04) |
| GPU prefix cache, still resident | About 40 ms | Only the new 500 tokens, attending to 32k |
| Evicted, reloaded from CPU | About 90 to 120 ms | Mostly exposed load (20.05) |
| Evicted, reloaded from NVMe | About 0.6 s | 7 GB/s tier |

<div class="take">
<div class="lab">Key takeaway</div>

a KV tier beyond GPU memory pays off when reusable KV exceeds HBM. Prove it with per-tier hit rates and per-turn TTFT, and check that it costs nothing when there is no reuse.

</div>
