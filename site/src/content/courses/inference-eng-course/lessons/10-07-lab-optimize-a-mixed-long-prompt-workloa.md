---
id: "10.07"
course: "inference-eng-course"
module: "10"
slug: "lab-optimize-a-mixed-long-prompt-workload"
title: "Lab: Optimize a Mixed Long-Prompt Workload"
keyIdeas: ["Workload: chat traffic plus long RAG prompts sharing a system prompt and some documents.", "Run four configs: neither feature, chunked prefill, prefix caching, both.", "Measure chat ITL p99, RAG TTFT p95, hit rate and goodput.", "Tune the chunk budget once both are on.", "Report each feature's contribution separately."]
question: "For a workload of chat plus long RAG prompts with a shared system prompt, which two features do you enable and what do you measure to prove each helped?"
answer: "Chunked prefill, proven by lower ITL p99 for chat users when long prompts arrive; and prefix caching, proven by the prefix hit rate metric and lower TTFT for requests with the shared system prompt or repeated documents. Measure each alone and together against the same workload and SLO."
sources: [{"title": "Sarathi-Serve", "url": "https://arxiv.org/abs/2403.02310", "note": "Chunked prefill."}, {"title": "vLLM: Automatic prefix caching design", "url": "https://docs.vllm.ai/en/latest/design/prefix_caching.html", "note": "Prefix caching."}]
tags: ["lab"]
tools: ["Lab"]
---

## Goal

Show, with measurements, what chunked prefill and prefix caching each do on a realistic mixed workload, and find the best combined configuration. The lab separates the two effects so you can explain each one.

## Workload

- **70% chat**: a shared 2,000-token system prompt plus a short user message, 300-token output.
- **30% RAG**: the same system prompt plus 6,000 tokens of documents, drawn from a pool of 50 documents with a skewed popularity (a few documents appear often), sorted by document id, 150-token output.
- **Arrivals**: open-loop Poisson at your expected rate, for at least 5 minutes after warm-up.
- **SLO**: for example p95 TTFT under 1 s, p95 TPOT under 40 ms and p99 ITL under 150 ms.

## The four configurations

| Config | Chunked prefill | Prefix caching |
| --- | --- | --- |
| A | off | off |
| B | on | off |
| C | off | on |
| D | on | on |

To turn chunked prefill off in recent vLLM, set the token budget at least as large as the longest prompt, so every prompt fits in one step. Keep every other setting identical.

## Expected pattern

| Metric | A | B | C | D |
| --- | --- | --- | --- | --- |
| Chat ITL p99 | bad | good | moderate | good |
| RAG TTFT p95 | high | higher | low | low |
| Prefix hit rate | 0 | 0 | high | high |
| Goodput | low | medium | medium | best |

Why:

- **A**: every RAG request runs an 8,000-token prefill in one step, stalling all chat streams.
- **B**: chunking removes the stalls, but every request still prefills its full prompt, so prefill uses a lot of compute and RAG TTFT rises.
- **C**: caching removes the system prompt from every prefill and popular documents from many. The remaining prefills are shorter, so stalls shrink, but uncached document prefills still stall streams.
- **D**: caching removes most prefill work, and chunking bounds what remains.

<div class="ex">
<div class="lab">Worked example</div>

**a set of results**

At 12 requests per second on an 8B model on an H100:

A: chat ITL p99 420 ms, RAG TTFT p95 900 ms, goodput 1,900 tok/s.

B: ITL p99 70 ms, RAG TTFT p95 1,300 ms, goodput 2,600 tok/s.

C: ITL p99 210 ms, RAG TTFT p95 260 ms, hit rate 71%, goodput 2,700 tok/s.

D: ITL p99 55 ms, RAG TTFT p95 310 ms, hit rate 71%, goodput 3,500 tok/s.

</div>

## Then tune

With configuration D, sweep the chunk budget over 1,024, 2,048 and 4,096. Pick the value with the highest goodput inside the SLO. Then run a rate sweep to find the new SLO-safe capacity and compare it with configuration A's.

## Explaining with metrics

For each configuration, report:

- the prefix hit rate from the engine's metrics,
- the distribution of step times, if you can log them, to show the long steps disappearing with chunking,
- the ITL time series for one chat user, which shows the stalls in A and C,
- prefill tokens per second actually computed, which shows caching's saving.

## Report

A table of the four configurations, the tuned result and the new capacity, plus one paragraph per feature that explains its effect with the metrics above.

## Deep dive: expected results before running the four configurations

Use the formulas from this module to predict the direction and rough size of each change before measuring:

| Configuration | Chat ITL p99 | RAG TTFT p95 | Why |
| --- | --- | --- | --- |
| A: no chunking, no caching | Stall-sized | Full prefill plus queue | Every long prompt stalls everyone (10.01) |
| B: chunking only | About one chunk step | Rises a little | Stall capped by the chunk size (10.02) |
| C: caching only | Falls, fewer long prefills | Falls by about ⟦m627⟧ times prefill | Less prefill to interfere (10.04) |
| D: both | Lowest | Low | Caching cuts prefill work, chunking caps what is left |

If a measurement goes the other way, check the hit rate first. A cache that never hits behaves exactly like configuration A or B.

<div class="take">
<div class="lab">Key takeaway</div>

chunked prefill protects decode latency, and prefix caching removes repeated prefill work. On mixed long-prompt traffic you want both, and measuring them separately is how you explain why.

</div>
