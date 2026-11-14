---
id: "30.06"
course: "inference-eng-course"
module: "30"
slug: "optimize-precision-kernels-runtime"
title: "Optimize Precision, Kernels & Runtime"
keyIdeas: ["Try FP8 or INT4 weights within the quality bar.", "Check attention backend and CUDA graph settings.", "Try speculative decoding if load is moderate.", "Gate every precision change on the quality eval.", "Keep the change log going."]
question: "How do you make sure a precision change in the capstone does not break the quality bar?"
answer: "Run the capstone quality eval (and format checks for tool calls) on the quantized configuration with production sampling settings and the same engine, compare against the BF16 reference with confidence intervals, and accept only if it stays above the bar set in 30.01. Also spot-check long-context and edge cases relevant to the product."
sources: [{"title": "lm-evaluation-harness", "url": "https://github.com/EleutherAI/lm-evaluation-harness", "note": "Evals."}, {"title": "Give Me BF16 or Give Me Death?", "url": "https://arxiv.org/abs/2411.02355", "note": "Quantization quality trade-offs."}]
tags: ["capstone"]
tools: ["Capstone"]
---

## Candidates

| Change | Gate |
| --- | --- |
| FP8 weights | quality eval |
| INT4 weights | quality eval, plus a check at large batch sizes |
| Attention backend | correctness, and speed at your shapes |
| CUDA graph sizes | startup time |
| Speculative decoding | the benefit depends on load (16.08) |

## Process

For each change, measure:

1. **Quality,** if the numerics change: rerun the eval and compare with the bar.
2. **SLO-safe capacity** on the workload file.
3. **Startup time and memory.**

<div class="ex">
<div class="lab">Worked example</div>

**FP8 weights for the support assistant**

Eval: 92.8%, down from 93.1%. Still above the 92% bar, but the margin is thin.

Capacity: 6.0 to 6.9 requests/s per GPU (+15%). Prefill speeds up from FP8 matmuls, and the smaller weights leave more room for KV.

Decision: adopt, and add a monthly eval run to watch the margin.

Speculation: at the production load of about 6 requests/s per GPU, running batch is around 30 to 40, and TPOT is already 24 ms against a 40 ms SLO. Speculation would cut TPOT but not raise capacity, which is limited by prefill. Not adopted.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

precision and runtime changes add capacity, but each must pass the quality gate. Speculation needs a check at real load.

</div>
