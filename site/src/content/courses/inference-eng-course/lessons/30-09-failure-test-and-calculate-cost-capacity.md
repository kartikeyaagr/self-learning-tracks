---
id: "30.09"
course: "inference-eng-course"
module: "30"
slug: "failure-test-and-calculate-cost-capacity"
title: "Failure-Test and Calculate Cost & Capacity"
keyIdeas: ["Run the chaos experiments from Module 24 on the capstone system.", "Fix weaknesses and re-test.", "Compute fleet size for peak with headroom and spares.", "Compute monthly cost and cost per request and per million tokens.", "Show sensitivity to traffic growth and mix changes."]
question: "What final capacity and cost numbers should the capstone report?"
answer: "Per-replica SLO-safe capacity for the workload; replicas needed at peak with headroom and failure spares; monthly GPU cost under the autoscaling policy and the resulting cost per million tokens and per conversation; and results of failure tests (recovery time, SLO impact) that justify the spare capacity."
sources: [{"title": "Principles of Chaos Engineering", "url": "https://principlesofchaos.org/", "note": "Failure testing."}, {"title": "Google SRE Book: Handling Overload", "url": "https://sre.google/sre-book/handling-overload/", "note": "Capacity and overload."}]
tags: ["capstone"]
tools: ["Capstone"]
---

## Failure tests

| Fault | Result | Recovery time | SLO impact |
| --- | --- | --- | --- |
| Replica killed | | | |
| Long-prompt burst | | | |
| Slow replica | | | |
| Retry storm | | | |

Use the method from 24.07: state the hypothesis, inject the fault, measure, fix and repeat.

## Capacity and cost

| Item | Value |
| --- | --- |
| Capacity per replica | |
| Peak replicas, with headroom | |
| Spares | |
| Average replicas, autoscaled | |
| Monthly GPU cost | |
| Cost per million output tokens | |
| Cost per conversation | |

<div class="ex">
<div class="lab">Worked example</div>

**the support assistant's numbers**

Capacity per replica: 6.9 requests/s. Peak: 9 replicas plus 1 spare = 10. Average with autoscaling: about 4.5 replicas, plus 1 warm.

Monthly GPU cost at \$3 per hour: 5.5 × \$3 × 730 ≈ \$12,000, well under the \$40,000 budget.

Output tokens per month: 15 requests/s × 220 tokens × 2.63 million seconds ≈ 8.7 billion. Cost per million output tokens: about \$1.40, with input processing included in that figure.

A conversation averages 4 turns, so it costs about 4 × \$12,000 ÷ (15 × 2.63 million) ≈ \$0.0012.

</div>

~~~dg-raw
%%widget cost Check the cost arithmetic with your own price, throughput and utilization.
p = 3
t = 3.3
u = 0.5
%%
~~~

## Sensitivity

Recompute for 50% more traffic, and for average prompts twice as long. Report the new replica counts and costs.

## Deep dive: sensitivity of the capacity plan

A capacity plan depends on estimates that will be wrong. Vary each input by a plausible error and see how the GPU count moves:

~~~dg-raw
%%bars GPUs needed for the support assistant when one input changes by its plausible error (baseline 12 GPUs, illustrative).
unit = GPUs
Peak traffic +30% | 16 | c1
Mean prompt length +50% | 14 | c2
Output length +25% | 14 | c2
Prefix-cache hit rate 70% → 40% | 15 | c1
TPOT SLO tightened 30 → 20 ms | 17 | c1
Baseline | 12 | c3
%%
~~~

The plan is most sensitive to the SLO and to peak traffic, so those deserve the most care in the defense, and the spare capacity should cover at least the largest single swing.

<div class="take">
<div class="lab">Key takeaway</div>

the capstone ends with numbers a business can use: capacity, cost and evidence that the system survives failures.

</div>
