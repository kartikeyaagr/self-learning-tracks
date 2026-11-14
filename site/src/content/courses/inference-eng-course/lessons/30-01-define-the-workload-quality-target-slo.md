---
id: "30.01"
course: "inference-eng-course"
module: "30"
slug: "define-the-workload-quality-target-slo"
title: "Define the Workload, Quality Target & SLO"
keyIdeas: ["Define the product and its traffic: lengths, sharing, rates, bursts.", "Write SLOs for TTFT, TPOT (or E2E) with percentiles and attainment.", "Set a measurable quality bar with an eval set.", "State constraints: budget, hardware availability, latency regions.", "Everything later in the capstone is judged against this."]
question: "Write a one-paragraph workload and SLO definition for a hypothetical support-chat product."
answer: "Support chat for a SaaS product: requests have a 2,500-token shared system prompt plus 500 to 3,000 tokens of conversation history and retrieved docs, outputs average 250 tokens (p95 600). Peak 80 requests per second, average 30, with 2x bursts. SLO: p95 TTFT under 800 ms, p95 TPOT under 40 ms, 99% attainment. Quality: at least 95% of the BF16 reference model's score on our 400-question support eval, and 100% valid JSON for tool calls."
sources: [{"title": "Google SRE Book: Service Level Objectives", "url": "https://sre.google/sre-book/service-level-objectives/", "note": "Writing SLOs."}, {"title": "DistServe", "url": "https://arxiv.org/abs/2401.09670", "note": "Goodput under SLOs."}]
tags: ["capstone"]
tools: ["Capstone"]
---

## The capstone

Over these ten lessons you design, build, optimize, harden and defend a complete inference system for one workload. Each lesson is one phase. A running example, a customer-support assistant, shows what each phase might produce. Use your own workload; the example only shows the level of detail to aim for.

## Definition template

| Item | Your value |
| --- | --- |
| Product | |
| Prompt length distribution | |
| Output length distribution | |
| Prefix sharing | |
| Peak and average rate | |
| Burst profile | |
| TTFT SLO | |
| TPOT or end-to-end SLO | |
| Attainment target | |
| Quality eval and bar | |
| Budget and constraints | |

<div class="ex">
<div class="lab">Worked example</div>

**the support assistant**

Product: a chat assistant that answers questions using a 3,000-token system prompt with policies, plus retrieved help articles.

Prompts: median 4,500 tokens, p95 9,000. Outputs: median 220 tokens, p95 600. Prefix sharing: the 3,000-token system prompt is in every request, and conversations average 4 turns.

Rate: 40 requests/s at peak, 15 on average. Bursts: up to 1.5 times the current rate within 5 minutes after product announcements.

SLO: TTFT p95 under 1.5 s and TPOT p95 under 40 ms, met by 99% of requests in every 5-minute window.

Quality: at least 92% correct on a 500-question internal eval, graded against reference answers. Budget: under \$40,000 per month in GPU cost.

</div>

## Build the workload file

Turn the definition into a replayable workload (Module 04): a file with arrival times, prompt lengths or text, and target output lengths. Every later phase replays this same file, so results stay comparable.

<div class="take">
<div class="lab">Key takeaway</div>

a precise workload, SLO and quality bar make every later decision checkable. Write them down before choosing anything.

</div>
