---
id: "16.08"
course: "inference-eng-course"
module: "16"
slug: "lab-map-where-speculative-decoding-wins"
title: "Lab: Map Where Speculative Decoding Wins"
keyIdeas: ["Compare speculation methods against a no-speculation baseline.", "Vary concurrency (1, 8, 32, 128), task type and temperature.", "Record TPOT, throughput, acceptance rate and mean accepted length.", "Produce a heatmap of gains and losses.", "Write a policy: when to enable which method."]
question: "Which variables define the map of where speculation helps, and what is the output?"
answer: "Method (draft model, n-gram, EAGLE-style), draft length, concurrency, task type (code, chat, RAG) and temperature. The output is a table or heatmap of TPOT and throughput change versus no speculation for each cell, plus acceptance rates, and a recommendation of which method and k to use per workload and load level."
sources: [{"title": "vLLM: Speculative decoding", "url": "https://docs.vllm.ai/en/latest/features/spec_decode.html", "note": "Configure methods."}, {"title": "EAGLE", "url": "https://arxiv.org/abs/2401.15077", "note": "One method to include."}]
tags: ["lab"]
tools: ["Lab"]
---

## Goal

Map where speculative decoding helps and where it hurts on your hardware and workloads, then turn the map into a serving policy. Speculation depends so strongly on task, load and sampling that a single benchmark number is misleading.

## The grid

| Axis | Values |
| --- | --- |
| Method | none, n-gram, draft model, EAGLE-style head |
| Draft length k | 2, 4, 8 |
| Concurrency | 1, 8, 32, 128 |
| Task | code editing, chat, RAG |
| Temperature | 0, 0.8 |

The full grid has hundreds of cells. If time is short, first find the best k per method at concurrency 1 on chat, then fix it and sweep the other axes.

## Measure

For each cell:

- p50 TPOT and p95 TPOT,
- output throughput,
- acceptance rate and mean accepted length,
- draft time and verification time per round, if your engine reports them.

Use fixed output lengths with `ignore_eos` where possible so methods are compared on the same work. For sampling at temperature 0.8, check that output quality is unchanged on a small eval, as a guard against implementation bugs.

## Expected shape

| | Concurrency 1 | Concurrency 32 | Concurrency 128 |
| --- | --- | --- | --- |
| n-gram, code editing | large win | win | small win |
| n-gram, chat | small win | none | small loss |
| EAGLE, chat | large win | win | near zero or a loss |
| Draft model, chat | win | small win | loss |

Temperature 0.8 should lower acceptance and shrink every win compared with greedy decoding.

<div class="ex">
<div class="lab">Worked example</div>

**a heatmap and the policy it produced**

On an 8B model on an H100, EAGLE with k = 4 cut chat TPOT by 2.3 times at concurrency 1, 1.6 times at 8, 1.15 times at 32, and lowered throughput by 4% at 128.

N-gram with k = 8 on code editing cut TPOT by 3.1 times at concurrency 1 and still by 1.4 times at 128, because drafts were free and acceptance stayed high.

Policy: "EAGLE with k = 4 when the running batch is below 32; n-gram with k = 8 on the code-editing endpoint at all loads; speculation off otherwise." SLO-safe capacity for chat rose 9%, and p50 TPOT at typical load fell 35%.

</div>

## Deliverable

- One heatmap per method, with concurrency on one axis and task on the other, coloured by TPOT speedup, with throughput change noted in each cell.
- A short table of acceptance rates by task and temperature.
- A one-paragraph policy that names the method, the draft length and the load threshold for each endpoint, with the measured effect on SLO-safe capacity.

## Deep dive: reading the sweep against the model

Fit ⟦m867⟧ from the measured mean accepted length ⟦m868⟧ at a fixed ⟦m869⟧ by solving

⟦M866⟧

numerically. If the fitted ⟦m870⟧ drifts with ⟦m871⟧, acceptance is not independent across positions (it usually falls with depth), and the policy should use measured accepted lengths per ⟦m872⟧ directly, not the formula.

<div class="take">
<div class="lab">Key takeaway</div>

speculation is workload and load dependent. A measured map across methods, tasks, loads and temperatures turns it from a gamble into a policy.

</div>
