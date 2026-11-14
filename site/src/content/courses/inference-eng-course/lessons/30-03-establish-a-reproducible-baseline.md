---
id: "30.03"
course: "inference-eng-course"
module: "30"
slug: "establish-a-reproducible-baseline"
title: "Establish a Reproducible Baseline"
keyIdeas: ["Default engine configuration with chosen model and GPU.", "Run the capstone workload across a rate sweep.", "Record goodput, SLO attainment, engine metrics.", "Measure run-to-run variance.", "Script everything so one command reproduces it."]
question: "What makes the capstone baseline good enough to measure all later optimizations against?"
answer: "It is fully pinned and scripted (one command reproduces it), uses the capstone workload file, measures the SLO metrics and goodput across a rate sweep, records engine and GPU metrics, and its run-to-run variance is known, so later gains can be judged against noise."
sources: [{"title": "vLLM: Engine arguments", "url": "https://docs.vllm.ai/en/latest/configuration/engine_args.html", "note": "Pinning configuration."}, {"title": "GuideLLM", "url": "https://github.com/vllm-project/guidellm", "note": "Rate sweeps."}]
tags: ["capstone"]
tools: ["Capstone"]
---

## Deliverables

- **baseline.yaml:** the model, engine version and every flag.
- **run_baseline.sh:** starts the engine, runs warm-up and the rate sweep, and writes results.
- **Results:** goodput against rate, TTFT and TPOT percentiles, engine metrics and an environment report (driver, CUDA, GPU clocks).
- **Variance:** 3 repeated runs at the expected rate.

## Baseline summary

| Metric | Value | Spread |
| --- | --- | --- |
| SLO-safe rate per GPU | | |
| TTFT p95 at the expected rate | | |
| TPOT p95 at the expected rate | | |
| KV usage at the expected rate | | |

<div class="ex">
<div class="lab">Worked example</div>

**support assistant baseline on one H100**

Engine defaults, BF16 weights and KV, prefix caching off.

SLO-safe rate: 3.1 requests/s per GPU (spread 3.0 to 3.2 over 3 runs). Above that, TTFT p95 crosses 1.5 s.

At 3 requests/s: TTFT p95 1.31 s, TPOT p95 24 ms, KV usage 41%.

TPOT is far inside the SLO and KV is under half full. TTFT is the limit. That already points at prefill as the bottleneck.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

a scripted, pinned baseline with known variance is the yardstick for the rest of the capstone.

</div>
