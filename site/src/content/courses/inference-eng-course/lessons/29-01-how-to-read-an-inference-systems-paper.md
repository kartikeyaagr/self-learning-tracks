---
id: "29.01"
course: "inference-eng-course"
module: "29"
slug: "how-to-read-an-inference-systems-paper"
title: "How to Read an Inference Systems Paper"
keyIdeas: ["Read in passes: abstract and figures, then method, then evaluation details.", "Identify the bottleneck the paper attacks and the workload where it matters.", "Check baselines: are they strong and current?", "Note hardware, model sizes, batch sizes, metrics and SLOs used.", "List claims you could test and assumptions that might not hold for you."]
question: "What three questions should you answer after a first pass over an inference systems paper?"
answer: "What problem and bottleneck does it target (and under what workload)? What is the core mechanism, in one or two sentences? What baseline and setup produced the headline number (hardware, model, batch size, metric), and does that setting match yours?"
sources: [{"title": "S. Keshav: How to Read a Paper", "url": "https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf", "note": "Three-pass reading method."}, {"title": "Efficient Memory Management for LLM Serving with PagedAttention", "url": "https://arxiv.org/abs/2309.06180", "note": "A good paper to practice on."}]
tags: ["research"]
---

## Three passes

| Pass | What to read | Goal |
| --- | --- | --- |
| 1 | title, abstract, figures, conclusion | what the paper claims and why it matters |
| 2 | method and key algorithms | how the technique works |
| 3 | evaluation and appendix | the conditions under which the result holds |

Most readers stop after the first pass. For systems papers, the third pass matters most, because the conditions decide whether the result applies to you.

## Questions for a systems paper

- **Which bottleneck** does it attack: memory, compute, communication or scheduling?
- **Which workload:** prompt and output lengths, arrival pattern, model size?
- **Which baseline:** a tuned, current engine, or an old version?
- **Which metric:** throughput, a latency percentile, or goodput under an SLO?
- **What does it cost:** memory, complexity, output quality?

<div class="ex">
<div class="lab">Worked example</div>

**applying the questions**

A paper claims "3.1× higher throughput" for a new scheduling method.

Bottleneck: prefill-decode interference. Workload: ShareGPT prompts, outputs capped at 512 tokens, arrivals all at time 0. Baseline: an engine version from a year earlier, without chunked prefill. Metric: throughput only, with no latency reported.

Today's engines already use chunked prefill, so much of the 3.1× may already be in your baseline. And with all requests arriving at once, the result says nothing about TTFT under realistic traffic.

</div>

## Red flags

- Baselines without continuous batching or paged KV.
- Throughput reported without latency.
- Only one model size or one type of hardware.
- No ablations, so you cannot tell which part of the technique matters.
- Workloads where every request arrives at once.

<div class="take">
<div class="lab">Key takeaway</div>

read for the bottleneck, the mechanism and the evaluation conditions. The conditions decide whether the result applies to you.

</div>
