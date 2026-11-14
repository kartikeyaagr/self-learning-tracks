---
id: "30.05"
course: "inference-eng-course"
module: "30"
slug: "optimize-memory-scheduling-caching"
title: "Optimize Memory, Scheduling & Caching"
keyIdeas: ["Apply memory tuning, scheduler tuning, chunked prefill and prefix caching.", "Change one thing at a time; keep a change log.", "Measure SLO-safe capacity after each change.", "Watch hit rate, KV usage and preemptions.", "Stop each line of work when it no longer moves capacity."]
question: "Which three optimizations from this phase are most likely to help a support-chat workload with a shared system prompt?"
answer: "Prefix caching (with sticky or prefix-aware routing) to skip the shared system prompt and conversation history; chunked prefill with a tuned token budget to keep TPOT stable when long histories arrive; and FP8 KV or memory tuning to raise concurrency without preemptions."
sources: [{"title": "vLLM: Automatic prefix caching design", "url": "https://docs.vllm.ai/en/latest/design/prefix_caching.html", "note": "Prefix caching."}, {"title": "Sarathi-Serve", "url": "https://arxiv.org/abs/2403.02310", "note": "Chunked prefill."}]
tags: ["capstone"]
tools: ["Capstone"]
---

## Order

1. **Prefix caching,** if prompts share prefixes.
2. **Chunked prefill and the token budget.**
3. **KV capacity:** the GPU memory fraction and FP8 KV.
4. **Maximum sequences.**

Each change is cheap to try and easy to revert. Make one change at a time and re-measure SLO-safe rate after each.

## Change log

| # | Change | SLO-safe rate | Change | Evidence |
| --- | --- | --- | --- | --- |
| 0 | Baseline | | | |
| 1 | Prefix caching | | | hit rate |
| 2 | Chunk budget | | | ITL p99 |
| 3 | FP8 KV | | | preemptions |
| 4 | Maximum sequences | | | TPOT |

<div class="ex">
<div class="lab">Worked example</div>

**the support assistant's change log**

0. Baseline: 3.1 requests/s per GPU.

1. Prefix caching: hit rate 71% of prompt tokens. 5.6 requests/s (+81%), close to the Amdahl estimate.

2. Chunk budget lowered from 8,192 to 2,048: ITL p99 falls from 85 ms to 38 ms. Rate unchanged at 5.6, but the TPOT tail now has room.

3. FP8 KV: KV usage at 5.6 requests/s falls from 74% to 37%; no preemptions. 6.0 requests/s (+7%).

4. Maximum sequences raised from 128 to 256: no change. The limit is now prefill compute, not sequence slots. Reverted.

Result: about 1.9 times the baseline capacity, with no change to model weights.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

memory, scheduling and caching changes are cheap and often give the largest gains. Measure each one against the baseline.

</div>
