---
id: "25.06"
course: "inference-eng-course"
module: "25"
slug: "project-build-the-cheapest-system-that-meets-the-slo"
title: "Project: Build the Cheapest System That Meets the SLO"
keyIdeas: ["Inputs: workload, SLO, quality bar, traffic forecast.", "Candidates: model sizes, precisions, GPUs, parallelism, engine features.", "Filter by quality, then measure SLO-safe capacity per GPU.", "Compute fleet size and monthly cost per candidate.", "Choose the cheapest; show sensitivity to traffic and mix changes."]
question: "Describe the steps to find the cheapest serving system for a given workload and SLO."
answer: "Fix the workload and SLO; shortlist models (sizes, quantizations) that pass quality; shortlist hardware by per-dollar ratios; benchmark each feasible combination for SLO-safe capacity; compute replicas needed for forecast peak with headroom and spares; compute monthly cost including idle and autoscaling assumptions; pick the cheapest, and document the runner-up and sensitivities."
sources: [{"title": "GuideLLM", "url": "https://github.com/vllm-project/guidellm", "note": "Benchmarking capacity."}, {"title": "lm-evaluation-harness", "url": "https://github.com/EleutherAI/lm-evaluation-harness", "note": "Quality gating."}]
tags: ["project"]
tools: ["Project"]
---

## Goal

Find the cheapest system that meets a stated SLO and quality bar, and show how the choice changes with traffic.

## Steps

1. **Requirements:** workload mix, SLO, quality threshold, peak and average traffic.
2. **Quality filter:** models and precisions that pass your evals (Module 11).
3. **Hardware shortlist:** per-dollar ratios (25.03).
4. **Benchmark:** SLO-safe capacity per replica for each combination.
5. **Fleet sizing:** replicas at peak and on average (25.04).
6. **Cost:** monthly GPU cost under your autoscaling assumptions.

## Result table

| Option | Model and precision | GPU | Replicas at peak | Monthly cost |
| --- | --- | --- | --- | --- |
| 1 | 8B FP8 | L40S | | |
| 2 | 8B FP8 | H100 | | |
| 3 | 14B INT4 | H100 | | |

<div class="ex">
<div class="lab">Worked example</div>

**how the table might come out**

Peak 60 requests/s, average 25. SLO: TTFT p95 under 1 s, TPOT p95 under 40 ms.

Option 1: capacity 4 requests/s per L40S, 22 replicas at peak, about 10 on average. At \$1.20 per hour, about \$8,800 per month.

Option 2: capacity 12 requests/s per H100, 8 at peak, about 4 on average. At \$3.00 per hour, about \$8,800 per month.

The two are tied on cost. Option 2 needs fewer machines to operate and has more room for longer prompts, which may decide it. These numbers are illustrative; your benchmarks decide.

</div>

## Sensitivity

Recompute for 50% more traffic and for prompts twice as long. The winner should still be reasonable. If it is not, note the risk in your report.

<div class="take">
<div class="lab">Key takeaway</div>

the cheapest system comes from combining quality gates, measured capacity and fleet math. Show the alternatives and how the choice changes with traffic.

</div>
