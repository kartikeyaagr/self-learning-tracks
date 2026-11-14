---
id: "19.06"
course: "inference-eng-course"
module: "19"
slug: "lab-profile-and-optimize-an-moe-workload"
title: "Lab: Profile and Optimize an MoE Workload"
keyIdeas: ["Serve an MoE model under TP and under EP (if available) on the same GPUs.", "Profile step time split: attention, experts, communication.", "Collect expert load statistics and imbalance.", "Measure goodput and TPOT for each configuration.", "Try one optimization: hot expert replication, batch size or backend."]
question: "What would you measure in an MoE profile to decide between more TP and more EP?"
answer: "The share of step time in MoE layers versus attention; per-expert GEMM sizes and efficiency; all-to-all (EP) versus all-reduce (TP) time; load imbalance across GPUs; and KV capacity per replica. If expert GEMMs are tiny and all-reduce heavy under TP, EP with larger per-expert batches may win; if all-to-all latency dominates at small scale, TP may be better."
sources: [{"title": "vLLM: Expert parallel deployment", "url": "https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html", "note": "EP options."}, {"title": "Mixtral of Experts", "url": "https://arxiv.org/abs/2401.04088", "note": "A model to test with."}]
tags: ["lab"]
tools: ["Lab"]
---

## Goal

Compare tensor parallelism and expert parallelism for one MoE model on your GPUs, profile both, and make one measured improvement.

## Steps

1. **Choose a model** that fits your GPUs: a Mixtral-class model or a smaller recent MoE such as Qwen MoE variants.
2. **Config A:** tensor parallelism across all GPUs.
3. **Config B:** expert parallelism for the MoE layers, with data-parallel or tensor-parallel attention, whichever your engine supports.
4. **Benchmark** a chat workload with your harness at three request rates: low, medium and near saturation.
5. **Profile** a steady-state decode step in each config.
6. **Log expert load:** tokens per expert per layer for a few hundred steps, if the engine exposes it.

## Profile table

| Component | Config A (TP) | Config B (EP) |
| --- | --- | --- |
| Attention | | |
| Expert GEMMs | | |
| All-reduce | | |
| All-to-all | | |
| Idle from imbalance | | |
| Step time | | |

## What to expect

- At low load, TP often has lower TPOT: all GPUs share every expert's read.
- At high load, EP often wins on throughput because expert GEMMs are larger.
- If max ÷ mean load in EP is above about 1.5, imbalance is eating the gain.

## Optimization

Pick one and measure goodput before and after:

- Raise the maximum batch size so expert GEMMs grow.
- Enable an optimized MoE kernel or communication backend.
- Replicate the hottest experts, if the engine supports it.

## Report

Write a short recommendation: TP or EP for this model and GPU count, backed by the profile, the load statistics and the goodput numbers.

## Deep dive: a predicted profile to check yours against

For one decode step of an MoE model, predict each part from bytes and messages before profiling:

| Part | Predicted from | What a large gap means |
| --- | --- | --- |
| Attention and shared layers | Shared weight bytes plus KV bytes, divided by bandwidth | Attention kernel or KV layout problem |
| Expert matmuls | Experts touched (19.01) × expert bytes, divided by bandwidth | Grouped GEMM inefficiency |
| All-to-all | Bytes per token × tokens, over the link, plus latency | Wrong transport, or no overlap |
| Router and gather | Small; launch-bound | Missing fusion or graph capture |

<div class="take">
<div class="lab">Key takeaway</div>

MoE performance is a balance of attention, small expert GEMMs and communication. Profile all three, and measure load balance, before choosing a parallel layout.

</div>
