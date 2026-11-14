---
id: "17.06"
course: "inference-eng-course"
module: "17"
slug: "project-build-an-engine-selection-matrix"
title: "Project: Build an Engine Selection Matrix"
keyIdeas: ["List must-haves: model, hardware, features.", "Measure goodput on two or three representative workloads per engine.", "Add operational criteria: startup, observability, upgrades, community.", "Weight criteria and score.", "Write a decision record with the data and the next-best option."]
question: "What columns and rows should an engine selection matrix have, and how do you turn it into a decision?"
answer: "Rows are candidate engines; columns are must-have requirements (model and hardware support, features), measured performance on your workloads (goodput under SLO, TTFT, TPOT), operational factors (startup time, metrics, deployment tooling, stability), and team factors (familiarity, ability to modify). Filter out engines failing must-haves, then weight the remaining columns by importance and pick the highest score, recording the reasoning."
sources: [{"title": "GuideLLM", "url": "https://github.com/vllm-project/guidellm", "note": "Benchmark client for the measurements."}, {"title": "vLLM benchmarks directory", "url": "https://github.com/vllm-project/vllm/tree/main/benchmarks", "note": "Alternative client."}]
tags: ["project"]
tools: ["Project"]
---

## Goal

Make the engine choice a documented, weighted decision based on your own workloads and constraints. The output is a selection matrix and a short decision record that anyone can review, and that tells you when to revisit the choice.

## The matrix

| Criterion | Weight | vLLM | SGLang | TensorRT-LLM |
| --- | --- | --- | --- | --- |
| Supports our model, precision and GPU | must have | | | |
| Goodput on the chat workload | 30% | | | |
| Goodput on the RAG workload | 20% | | | |
| p95 TTFT at the expected load | 15% | | | |
| Startup time | 10% | | | |
| Metrics and operational tooling | 15% | | | |
| Team familiarity | 10% | | | |

Adjust the criteria and weights to your situation. A latency-critical product might weight TPOT heavily. A team with frequent model changes might weight model support and setup time. Agree on the weights before you see results, so they are not adjusted to favour a preferred engine.

## The process

1. **Filter by must-haves.** Drop any engine that cannot run your model, precision, context length or hardware, or lacks a feature you depend on, such as LoRA serving or a specific structured-output mode.
2. **Benchmark fairly.** Use the method from lesson 17.01, with equal tuning effort, on each workload in the matrix. Add isolation tests from lesson 17.05 where they inform a criterion.
3. **Score each criterion from 1 to 5.** For measured criteria, map results to scores with a rule, such as 5 for the best and points off for each 5% behind. For qualitative ones, write a sentence of justification.
4. **Compute the weighted sum.**
5. **Sanity check the winner.** Does it have a weakness the matrix missed? Would a small change in weights flip the result? If so, say so.
6. **Write the decision record.**

<div class="ex">
<div class="lab">Worked example</div>

**a filled-in result**

After filtering, two engines remained. Weighted totals: engine X 4.1, engine Y 3.8.

X led on chat goodput (+9%) and tooling. Y led on RAG goodput (+6%) and TTFT.

A sensitivity check showed that raising the RAG weight from 20% to 35% would flip the result. The record noted this and named the trigger: "if RAG grows above about a third of traffic, re-evaluate."

</div>

## The decision record

Half a page, containing:

- **The winner** and its weighted score, with the runner-up's.
- **The key numbers** behind the decision, linked to the benchmark results.
- **Risks**: weaknesses of the winner and how you will mitigate them.
- **When to revisit**: a new major version of any candidate, a new model architecture, a significant change in workload mix, or a fixed date such as six months.

## Why bother

Engine choices tend to be made once and then defended by habit. A written matrix makes the reasons explicit, so when circumstances change, the team can see whether the original reasons still hold. It also turns an argument about preferences into a discussion about weights and measurements, which is easier to settle.

## Deep dive: a decision with uncertainty

Score each option on criteria with weights ⟦m892⟧, and carry the uncertainty of each score:

⟦M891⟧

If two options' scores differ by less than about twice the combined SE, the record should say so and decide on the criterion that matters most, usually operational risk, not on the total.

<div class="take">
<div class="lab">Key takeaway</div>

make the engine choice a documented, weighted decision based on your own workloads, with must-haves filtered first and weights agreed in advance. Record when to revisit it, because versions, models and traffic all change.

</div>
