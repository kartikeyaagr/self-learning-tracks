---
id: "30.10"
course: "inference-eng-course"
module: "30"
slug: "publish-and-defend-the-complete-system"
title: "Publish and Defend the Complete System"
keyIdeas: ["Write a design document: requirements, decisions, results, risks.", "Include the change log, charts and decision records.", "Publish code, configs and workload files.", "Present and take questions on every decision.", "List limitations and next steps honestly."]
question: "What questions should you be ready to answer when defending your capstone system?"
answer: "Why this model, precision, GPU and engine (with evidence); what the binding constraint was and how each optimization moved it; why you did or did not use multi-GPU or disaggregation; how the system behaves under bursts and failures; what it costs per request and how that scales; what quality risks remain; and what you would do next with more time or budget."
sources: [{"title": "S. Keshav: How to Read a Paper", "url": "https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf", "note": "Reviewers read your work the same way."}, {"title": "Google SRE Book: Postmortem Culture", "url": "https://sre.google/sre-book/postmortem-culture/", "note": "Writing honestly about failures."}]
tags: ["capstone"]
tools: ["Capstone"]
---

## Design document outline

1. Workload, SLO and quality bar.
2. Model, hardware and engine decisions, with evidence.
3. Baseline.
4. Bottleneck analysis.
5. Optimizations: the change log with gains.
6. Scaling decision.
7. Routing, autoscaling and observability.
8. Failure testing.
9. Capacity and cost.
10. Risks, limitations and next steps.

## Defense questions to prepare

| Topic | Question |
| --- | --- |
| Evidence | What would change your engine choice? |
| Bottleneck | What limits capacity now? |
| Scaling | Why not tensor parallelism or disaggregation? |
| Reliability | What happens if a node dies at peak? |
| Cost | How does cost change with twice the traffic? |
| Quality | Where might quantization hurt users? |

<div class="ex">
<div class="lab">Worked example</div>

**answering "What limits capacity now?"**

"Prefill compute. After prefix caching, each request still prefills about 1,500 uncached tokens. At 6.9 requests/s that is about 10,000 tokens/s of prefill, plus decode, and the TTFT p95 SLO is what breaks first as rate rises."

"The next lever is a higher prefix hit rate, by caching retrieved articles that many users ask about. After that, FP8 attention, or disaggregation once the fleet is large enough to justify it."

A good answer names the constraint, gives the number behind it and says what you would try next.

</div>

## Deep dive: the formula sheet

The equations that answer most questions in a design review:

| Question | Formula | Module |
| --- | --- | --- |
| Does it fit? | ⟦m1164⟧ | 01.01 |
| KV per token | ⟦m1165⟧ | 01.01 |
| Prefill time | ⟦m1166⟧ | 03.02 |
| Decode step | ⟦m1167⟧ | 03.03 |
| Throughput | ⟦m1168⟧ | 05.02 |
| Waiting | ⟦m1169⟧, ⟦m1170⟧ | 04.02 |
| Goodput | requests meeting every SLO per second | 04.06 |
| Ridge | ⟦m1171⟧, about 295 FLOP/byte on H100 BF16 | 12.07 |
| Speculation | ⟦m1172⟧ tokens per round | 16.02 |
| All-reduce | ⟦m1173⟧ | 18.02 |
| Cost | ⟦m1174⟧ per M tokens | 25.01 |
| Any speed-up | ⟦m1175⟧ | 06.06 |

If a reviewer asks a number and you can place it in one of these rows, you can answer with the arithmetic in front of them.

<div class="take">
<div class="lab">Key takeaway</div>

the final deliverable is a system plus the evidence for every decision in it. Being able to defend each choice with data is what inference engineering is.

</div>
