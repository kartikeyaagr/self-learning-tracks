---
id: "30.07"
course: "inference-eng-course"
module: "30"
slug: "decide-on-multi-gpu-or-disaggregation"
title: "Decide on Multi-GPU or Disaggregation"
keyIdeas: ["Default: single-GPU replicas if the SLO is met.", "TP when per-replica latency or KV capacity is insufficient.", "Disaggregation when interference is the binding problem at scale.", "Compare options on goodput per GPU and per dollar.", "Document why you did or did not scale out."]
question: "Under what results from earlier phases would you move the capstone to multi-GPU replicas or disaggregation?"
answer: "Move to tensor parallelism if single-GPU TPOT cannot meet the SLO or KV capacity is insufficient after other optimizations. Consider disaggregation if long prompts cause decode interference that chunked prefill cannot fix at the required load and the network between GPUs is fast. Otherwise stay with single-GPU replicas, which are simplest and usually most efficient."
sources: [{"title": "vLLM: Parallelism and scaling", "url": "https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html", "note": "Parallel options."}, {"title": "DistServe", "url": "https://arxiv.org/abs/2401.09670", "note": "Disaggregation trade-offs."}]
tags: ["capstone"]
tools: ["Capstone"]
---

## Decision tree

~~~dg-flow Scaling decision.
Meets SLO on 1 GPU? -(yes)-> *Replicas (data parallel)
Meets SLO on 1 GPU? -(no: TPOT or KV)-> Tensor parallel 2 -> Re-measure
Interference dominates at scale? -(yes, fast network)-> Prefill-decode disaggregation
~~~

## Compare

| Option | Goodput per GPU | Cost per million tokens | Complexity |
| --- | --- | --- | --- |
| 1-GPU replicas | | | low |
| TP 2 replicas | | | medium |
| Disaggregated | | | high |

<div class="ex">
<div class="lab">Worked example</div>

**the support assistant's choice**

One H100 meets the SLO at 6.9 requests/s. Peak is 40 requests/s. At 70% target utilization, that is ceil(40 ÷ 4.8) = 9 replicas.

TP 2 would cut TPOT, which already has plenty of room, and would not raise goodput per GPU.

Disaggregation could help: prompts are long and outputs short, so a split of about 3 prefill and 4 decode GPUs might fit (see the widget below). But ITL is already within the SLO, and the gain would be modest for much more complexity.

Decision: 1-GPU replicas. Revisit disaggregation if traffic grows past about 30 GPUs.

</div>

~~~dg-raw
%%widget pools Check the support assistant's ratio: 40 requests/s, 4,500-token prompts, 220-token outputs, 71% prefix hit rate.
r = 40
p = 12.15
n = 7.8
h = 0.7
%%
~~~

<div class="take">
<div class="lab">Key takeaway</div>

scale out only when single-GPU replicas cannot meet the SLO. Compare options on goodput per dollar and pick the simplest that works.

</div>
