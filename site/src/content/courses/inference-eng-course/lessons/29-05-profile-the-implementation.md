---
id: "29.05"
course: "inference-eng-course"
module: "29"
slug: "profile-the-implementation"
title: "Profile the Implementation"
keyIdeas: ["Profile baseline and technique on the same workload.", "Check that the claimed mechanism shows up (bytes, idle time, kernel counts).", "Find new bottlenecks the technique introduced.", "Relate the gain to hardware ratios (bandwidth, FLOPs, interconnect).", "Use the profile to predict where the technique stops working."]
question: "Why profile a technique after reproducing its speedup?"
answer: "To confirm the speedup comes from the mechanism the paper claims (for example fewer bytes moved or less idle time) rather than an unrelated side effect, to find where the remaining time goes, and to understand which hardware or workload properties the gain depends on, which predicts where it will fail."
sources: [{"title": "NVIDIA Nsight Systems User Guide", "url": "https://docs.nvidia.com/nsight-systems/UserGuide/index.html", "note": "Timeline profiling."}, {"title": "NVIDIA Nsight Compute documentation", "url": "https://docs.nvidia.com/nsight-compute/NsightCompute/index.html", "note": "Kernel profiling."}]
tags: ["research", "profiling"]
---

## What to compare

| Item | Baseline | Technique |
| --- | --- | --- |
| Step time breakdown | | |
| Kernel count per step | | |
| GPU idle share | | |
| Bytes moved by key kernels | | |
| Communication time | | |

## Confirm the mechanism

The profile should show the mechanism the paper claims.

- If the paper claims it "removes prefill stalls", the step-time trace should show steps with a bounded maximum length.
- If it claims it "reduces KV reads", decode attention should move fewer bytes.
- If it claims "better overlap", the gaps between kernels should shrink.

<div class="ex">
<div class="lab">Worked example</div>

**confirming chunked prefill**

Baseline trace: most steps take 12 to 14 ms, but every few seconds a step takes 180 ms when a long prompt is prefilled in one go.

With chunked prefill and a 512-token budget: no step takes more than about 30 ms. The 180 ms of prefill work is spread over about eight steps.

That matches the claim. The trace also shows the total time of those eight steps is slightly more than 180 ms, because each chunk re-reads the KV of earlier chunks. That is the technique's cost.

</div>

## New bottlenecks

Techniques often move the bottleneck rather than remove it. After chunked prefill, the next limit might be scheduler overhead or attention re-reading earlier chunks. Note the new bottleneck in your report; it is where the next improvement will come from.

<div class="take">
<div class="lab">Key takeaway</div>

profiling shows why a technique works. It also shows what it costs and where it will stop helping.

</div>
