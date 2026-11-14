---
id: "29.08"
course: "inference-eng-course"
module: "29"
slug: "project-compare-against-modern-engines-publish"
title: "Project: Compare Against Modern Engines & Publish"
keyIdeas: ["Compare the technique against a current engine with its best settings.", "Many older paper gains are now built into engines; quantify what remains.", "Publish code, workload files and configs.", "Write clearly: claim, setup, results, limits.", "Share for review and update with feedback."]
question: "What should a published reproduction report contain to be useful to others?"
answer: "The paper and claim reproduced; full setup (hardware, versions, flags, workload file); baseline match; main result with spread; profile showing the mechanism; ablations; the map of where it helps and stops helping; a comparison against a current engine's default (does the gap still exist?); and code and data to rerun it."
sources: [{"title": "ML Reproducibility Challenge", "url": "https://reproml.org/", "note": "Report structure."}, {"title": "vLLM documentation", "url": "https://docs.vllm.ai/en/latest/", "note": "Modern baseline."}]
tags: ["project"]
tools: ["Project"]
---

## Compare with modern engines

| Config | Metric |
| --- | --- |
| Paper baseline (old engine) | |
| Paper technique | |
| Current engine, defaults | |
| Current engine, tuned | |
| Current engine plus the technique (if it applies) | |

The most useful question is often not "does the technique work?" but "does it still matter?" A current engine may already include it, or something that gives the same benefit.

## Report outline

1. **Claim** and a short summary of your findings.
2. **Setup and workload,** with the workload files attached.
3. **Baseline match:** your baseline against the paper's.
4. **Main result** with variance across runs.
5. **Mechanism:** the profile that shows why it works.
6. **Ablations:** which parts matter.
7. **Where it stops working:** the gain map and its explanation.
8. **Comparison with current engines.**
9. **Limitations:** what you could not match, and how that might affect the conclusion.

<div class="ex">
<div class="lab">Worked example</div>

**a one-paragraph summary**

"We reproduced the paper's main result partially: 1.6× throughput (1.5 to 1.7 over 3 runs) against the paper's 2.3×, on an H100 instead of an A100. Most of the difference comes from a stronger baseline: the current engine's chunked prefill already removes some of the stalls the technique targets. The scheduling policy accounts for about 90% of the gain; the custom kernel adds little. Gains fall below 1.1× for prompts under 1,000 tokens."

</div>

<div class="take">
<div class="lab">Key takeaway</div>

a good reproduction says whether a result still matters with today's engines, and gives others everything they need to check it.

</div>
