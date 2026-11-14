---
id: "07.07"
course: "inference-eng-course"
module: "07"
slug: "lab-tune-ttft-vs-throughput"
title: "Lab: Tune TTFT vs Throughput"
keyIdeas: ["Choose a realistic mixed workload and an SLO for TTFT and TPOT.", "Sweep token budget, then max sequences, then load.", "Plot each setting's TTFT p95 vs TPOT p95 at fixed load: a trade-off frontier.", "Pick the setting with the most goodput under the SLO.", "Record why the winner wins using engine metrics."]
question: "You must hit p95 TTFT under 800 ms and p95 TPOT under 40 ms at the highest possible rate. Which knobs do you sweep and in what order?"
answer: "First set max_num_batched_tokens (with chunked prefill on) to the smallest value meeting the TTFT target at the expected load. Then raise max_num_seqs until TPOT p95 approaches 40 ms. Then increase request rate until either target breaks, and record that as capacity. Re-check prefix caching and gpu_memory_utilization if KV becomes the limit."
sources: [{"title": "vLLM: Optimization and tuning", "url": "https://docs.vllm.ai/en/latest/configuration/optimization.html", "note": "The relevant flags."}, {"title": "GuideLLM", "url": "https://github.com/vllm-project/guidellm", "note": "Rate sweeps with SLO views."}]
tags: ["lab"]
tools: ["Lab"]
---

## Goal

Find the scheduler configuration that delivers the most goodput inside your SLO, for a realistic mixed workload, and explain why it wins. This lab applies everything in this module to a real engine.

## Setup

- **Workload**: a mix such as 70% chat (500 tokens in, 300 out) and 30% RAG (4,000 tokens in, 150 out), generated from a seeded file.
- **SLO**: p95 TTFT under 800 ms and p95 TPOT under 40 ms, with 99% attainment.
- **Tools**: your harness from Module 04 in open-loop mode, and the engine's metrics scraped every 5 seconds.
- **Engine**: vLLM or SGLang with chunked prefill on, all other flags written down.

## Sweep plan

Change one knob at a time, in order of expected impact.

1. **Token budget.** Fix the rate at a moderate level, about 70% of the expected capacity. Sweep max_num_batched_tokens over 512, 1,024, 2,048, 4,096 and 8,192.
2. **Max sequences.** At the best budget, sweep max_num_seqs over 64, 128, 256 and 512.
3. **Rate.** At the best pair, raise the request rate in steps of about 10% until attainment drops below 99%.

Run each point for at least 3 minutes after warm-up, and repeat the most important points to check variance.

## Plot the results

Make a scatter plot with p95 TTFT on the x-axis and p95 TPOT on the y-axis. Each point is one configuration, labelled with its goodput. Draw the SLO as a box in the lower-left corner. The best configuration is the point inside the box with the highest goodput.

| Budget | Max seqs | p95 TTFT (ms) | p95 TPOT (ms) | Goodput (tok/s) |
| --- | --- | --- | --- | --- |
| 512 | 256 | 1,400 | 31 | 2,900 |
| 2,048 | 256 | 650 | 36 | 4,100 |
| 8,192 | 256 | 420 | 52 | 3,100 |

In this illustrative table, the 512 budget fails TTFT because long RAG prompts take too many steps to prefill. The 8,192 budget fails TPOT because long prefill steps slow every decode. The 2,048 budget sits inside the box.

## Explaining the winner with metrics

A configuration that wins without explanation might win by luck. Use the engine's metrics to show why:

- **Waiting queue.** Low and stable for the winner. Growing for a configuration that is under-admitting.
- **KV usage.** Below about 90% for the winner. Near 100% with preemptions for a configuration whose max_num_seqs is too high.
- **Step time distribution.** If you log it, the winner should have few very long steps.
- **ITL tail.** p99 ITL tells you about prefill stalls that p95 TPOT can hide.

<div class="ex">
<div class="lab">Worked example</div>

**a write-up paragraph**

"Budget 2,048 with 256 max sequences serves 9.5 requests per second at 99.2% SLO attainment. At budget 8,192, steps containing a full RAG prefill take up to 140 ms, pushing p95 TPOT to 52 ms. At budget 512, a 4,000-token prompt needs 9 steps to prefill, and under load it waits behind others, pushing p95 TTFT to 1.4 s. The waiting queue at the chosen setting averages 1.3 requests and KV usage peaks at 81%, so neither memory nor admission is limiting."

</div>

## Deliverable

The scatter plot, the sweep tables, the chosen configuration with its SLO-safe rate, and a paragraph that explains the result using engine metrics. Add the configuration to your baseline file so later modules start from it.

## Deep dive: reading a Pareto front

Each configuration gives a point (throughput at the SLO, latency). A configuration is dominated if another one is at least as good on both axes and better on one. Only undominated points are worth discussing.

~~~dg-raw
%%chart p95 TPOT against request rate for three token budgets (illustrative lab result).
x = 4, 6, 8, 10, 12
xlab = Requests per second
ylab = p95 TPOT (ms)
Budget 512: 18, 21, 26, 38, 95
Budget 2,048: 20, 23, 28, 36, 52
Budget 8,192: 27, 33, 45, 70, 140
SLO, 40 ms*: 40, 40, 40, 40, 40
%%
~~~

With a 40 ms TPOT SLO, budget 2,048 stays inside it up to about 10.5 req/s, budget 512 up to about 10, and budget 8,192 up to about 7. Budget 8,192 is dominated everywhere. The choice between 512 and 2,048 depends on whether TTFT, which the smaller budget raises, has its own SLO.

<div class="take">
<div class="lab">Key takeaway</div>

tuning is a search for the most goodput inside the SLO box. Sweep one knob at a time in order of impact, plot TTFT against TPOT, and explain the winner with the engine's own metrics.

</div>
