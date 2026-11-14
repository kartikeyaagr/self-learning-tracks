---
id: "30.02"
course: "inference-eng-course"
module: "30"
slug: "choose-the-model-hardware-engine"
title: "Choose the Model, Hardware & Engine"
keyIdeas: ["Shortlist models that can meet the quality bar.", "Match hardware to the workload's bottleneck and the model's memory.", "Choose the engine with a fair comparison and selection matrix.", "Record each decision with evidence and the runner-up.", "Keep options open for later phases (precision, parallelism)."]
question: "What evidence should support your model, hardware and engine choices?"
answer: "Model: eval scores against the quality bar for each candidate size and precision. Hardware: memory fit and per-dollar ratios matched to whether the workload is prefill or decode heavy, plus a quick benchmark. Engine: a fair comparison on your workload (goodput under SLO) and required features, recorded in a selection matrix."
sources: [{"title": "lm-evaluation-harness", "url": "https://github.com/EleutherAI/lm-evaluation-harness", "note": "Quality evaluation."}, {"title": "GuideLLM", "url": "https://github.com/vllm-project/guidellm", "note": "Benchmarking."}]
tags: ["capstone"]
tools: ["Capstone"]
---

## Decisions

| Decision | Evidence | Lesson |
| --- | --- | --- |
| Model and size | eval scores | 11.07 |
| Initial precision | eval scores and memory | 11.08 |
| GPU type | per-dollar ratios plus a quick benchmark | 25.03 |
| Engine | a fair comparison and the features you need | 17.06 |

## Decision record

For each decision, write down:

- the option chosen, with its numbers,
- the runner-up, and why it lost,
- what would make you revisit the decision.

<div class="ex">
<div class="lab">Worked example</div>

**model decision record**

Chosen: an 8B instruct model. Eval: 93.1% on the internal set, above the 92% bar.

Runner-up: a 70B model at 96.4%. It lost because it needs about 6 times the GPUs for the same traffic, and 93.1% already passes the bar.

Revisit if the quality bar rises above 94%, or if user feedback shows failure types the eval does not cover.

</div>

## Choosing the GPU

This workload has long prompts and short outputs, so prefill matters more than usual. FLOPs per dollar and memory for the 3,000-token cached prefix both count.

~~~dg-raw
%%widget gpuval Compare FLOPs per dollar and bandwidth per dollar. A prefill-heavy workload leans on the first; a chat workload on the second.
%%
~~~

<div class="take">
<div class="lab">Key takeaway</div>

choose the model, hardware and engine on measured evidence, and write the reasoning down so you can revisit it.

</div>
