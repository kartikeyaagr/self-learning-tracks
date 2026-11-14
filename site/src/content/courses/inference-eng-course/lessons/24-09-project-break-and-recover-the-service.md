---
id: "24.09"
course: "inference-eng-course"
module: "24"
slug: "project-break-and-recover-the-service"
title: "Project: Break and Recover the Service"
keyIdeas: ["Run your production-shaped service under load.", "Inject at least five faults from the chaos list.", "Measure impact and recovery time for each.", "Fix at least two weaknesses and re-test.", "Write runbook entries for each failure."]
question: "What should the final report of this project contain?"
answer: "The service setup and SLO; a list of injected faults with hypotheses; for each, observed impact (SLO attainment, errors, recovery time) with charts; the weaknesses found and fixes applied; re-run results after fixes; and a runbook entry for each failure type."
sources: [{"title": "Principles of Chaos Engineering", "url": "https://principlesofchaos.org/", "note": "Experiment design."}, {"title": "Google SRE Book: Postmortem Culture", "url": "https://sre.google/sre-book/postmortem-culture/", "note": "Writing up incidents."}]
tags: ["project"]
tools: ["Project"]
---

## Goal

Break your own service on purpose, measure what happens, fix what breaks, and write a runbook entry for each fault.

## Plan

| # | Fault | Hypothesis | Measure |
| --- | --- | --- | --- |
| 1 | Kill a replica | recovery in under 30 s | errors and SLO attainment per minute |
| 2 | Long-prompt burst | no OOM | ITL p99, crashes |
| 3 | Client disconnect storm | KV is freed | KV usage, running sequences |
| 4 | Slow replica | traffic routes around it | traffic per replica |
| 5 | Retry storm | backoff holds | request multiplier |

## For each fault

1. Run steady load from your harness.
2. Inject the fault and mark its start and end on the dashboard.
3. Chart SLO attainment and errors over time.
4. Compute time to detect and time to recover.
5. If the hypothesis failed, fix the cause and run the experiment again.

## Runbook entry template

- **Symptom:** what the on-call engineer sees first.
- **Likely cause:** what usually triggers it.
- **Dashboards to check:** links, in order.
- **Mitigation steps:** what to do in the first 10 minutes.
- **Follow-up:** what to fix so it does not recur.

<div class="take">
<div class="lab">Key takeaway</div>

breaking your own service on purpose, and fixing what breaks, is the fastest way to make it reliable.

</div>
