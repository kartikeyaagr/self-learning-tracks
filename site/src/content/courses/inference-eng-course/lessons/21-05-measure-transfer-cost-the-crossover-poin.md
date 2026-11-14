---
id: "21.05"
course: "inference-eng-course"
module: "21"
slug: "measure-transfer-cost-the-crossover-point"
title: "Measure Transfer Cost & the Crossover Point"
keyIdeas: ["Disaggregation adds transfer latency and coordination overhead.", "It wins when interference is the main SLO problem: long prompts, high load, tight TPOT.", "Colocated with chunked prefill is a strong baseline; compare against it, not naive colocation.", "Find the crossover: load or prompt length where disaggregation starts to win goodput.", "Measure with the same total GPUs in both setups."]
question: "Under what conditions does colocated serving with chunked prefill beat disaggregation?"
answer: "At low load, where interference is small and disaggregation's fixed overheads (KV transfer, extra hop, underused separate pools) dominate; with short prompts, where prefill is cheap and interference minor; when the network between pools is slow; or when traffic's prefill/decode ratio swings faster than pools can be rebalanced, leaving one pool idle."
sources: [{"title": "DistServe", "url": "https://arxiv.org/abs/2401.09670", "note": "Goodput comparison."}, {"title": "Sarathi-Serve", "url": "https://arxiv.org/abs/2403.02310", "note": "Strong colocated baseline with chunked prefill."}]
tags: ["disaggregation"]
---

## A fair comparison

Compare with the same total GPUs:

- **Colocated:** N replicas with chunked prefill and a tuned token budget.
- **Disaggregated:** P prefill GPUs plus D decode GPUs with P + D = N, using the best split.

Comparing disaggregation against an untuned colocated baseline will always make disaggregation look good.

## Sweep

Vary the request rate and the prompt length. At each point, measure goodput: requests per second that meet the SLO.

| Condition | Usually better |
| --- | --- |
| Low load | colocated |
| Short prompts | colocated |
| Long prompts, high load, tight TPOT SLO | disaggregated |
| Slow network between pools | colocated |
| Very bursty shifts in traffic mix | colocated, or workers that can switch roles |

## The crossover chart

Plot goodput against request rate for both setups on the same chart. The rate where the lines cross is where disaggregation starts to pay. Report it for two or three prompt lengths.

<div class="ex">
<div class="lab">Worked example</div>

**reading a crossover chart**

On 8 GPUs with 4k prompts and a 40 ms TPOT SLO, colocated serving might meet the SLO up to 30 requests/s, then degrade fast as prefill stalls push p95 TPOT over 40 ms.

A 3 prefill + 5 decode split might reach only 26 requests/s at low load, because some GPUs sit partly idle, but hold the SLO up to 38 requests/s.

The crossover is around 30 requests/s. Below it, colocate. Above it, disaggregate. With 1k prompts, the crossover may never come.

</div>

## Deep dive: why the crossover depends on the SLO

Colocation shares every GPU between both phases, so it uses hardware fully when SLOs are loose. Disaggregation dedicates GPUs, so it wastes some capacity, but it keeps TPOT flat under prefill load, so it can run decode closer to a tight SLO.

~~~dg-raw
%%chart Goodput per GPU against the TPOT SLO (illustrative crossover for a RAG-like mix).
x = 20, 30, 50, 80, 120
xlab = TPOT SLO (ms)
ylab = Goodput (tokens/s per GPU)
Colocated: 600, 1400, 2300, 2900, 3200
Disaggregated: 1500, 2100, 2500, 2700, 2800
%%
~~~

Below about 50 ms, disaggregation wins here; above it, colocation does. The crossover moves left for chat-like traffic (little prefill to interfere) and right for prompt-heavy traffic.

<div class="take">
<div class="lab">Key takeaway</div>

disaggregation is not free. Compare it against tuned colocated serving on the same GPUs, and find the load and prompt length where it starts to win.

</div>
