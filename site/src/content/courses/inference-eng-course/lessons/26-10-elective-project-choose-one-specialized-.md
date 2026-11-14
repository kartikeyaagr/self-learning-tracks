---
id: "26.10"
course: "inference-eng-course"
module: "26"
slug: "elective-project-choose-one-specialized-workload"
title: "Elective Project: Choose One Specialized Workload"
keyIdeas: ["Choose one workload from this module.", "Define traffic shape and success metric (TTFT, task time, cost per image).", "Baseline on a standard setup.", "Apply two or three targeted optimizations.", "Report gains with quality checks."]
question: "How do you scope an elective project on a specialized workload so it produces a clear result?"
answer: "Pick one workload (for example multi-LoRA, structured output, VLM, agents, reasoning, batch, RL rollouts or diffusion), define its realistic traffic and metric of success, build a baseline on a standard engine, apply two or three targeted optimizations from the lesson, and report the measured gain against the baseline with quality checks."
sources: [{"title": "vLLM documentation", "url": "https://docs.vllm.ai/en/latest/", "note": "Features for most specialized workloads."}, {"title": "Hugging Face Diffusers", "url": "https://huggingface.co/docs/diffusers/index", "note": "For the diffusion option."}]
tags: ["project"]
tools: ["Project"]
---

## Options and example targets

| Workload | Example metric | Example optimizations |
| --- | --- | --- |
| Structured output | TPOT overhead, validity | grammar backend, mask caching |
| Multi-LoRA | adapters per GPU at the SLO | adapter pool sizing, rank |
| Vision-language | TTFT with images | encoder cache, resolution |
| Agents | task completion time | prefix caching, sticky routing, KV retention |
| Reasoning | cost per solved task | speculation, thinking budget |
| Batch | tokens per dollar | batch size, prefix grouping |
| RL rollouts | rollout time per step | parallel sampling, tail handling |
| Diffusion | latency at a quality bar | steps, quantization |

## Scoping

Pick one workload and one metric. A good project changes two or three things and measures each separately, so you know which change helped.

## Deliverable

1. **Workload definition:** what traffic looks like, with numbers.
2. **Baseline:** the metric before any change, with the configuration recorded.
3. **Changes:** each one described and measured on its own.
4. **Results table:** metric, latency and cost for each step.
5. **Quality check:** evidence that output quality did not regress.
6. **What generalizes:** one paragraph on which findings apply to other workloads.

<div class="take">
<div class="lab">Key takeaway</div>

specialized workloads reward targeted optimizations. Scope tightly, measure the right metric and compare against a clean baseline.

</div>
