---
id: "27.06"
course: "inference-eng-course"
module: "27"
slug: "project-modify-and-profile-one-engine-component"
title: "Project: Modify and Profile One Engine Component"
keyIdeas: ["Pick one component: scheduler policy, eviction policy, a kernel, sampling, metrics.", "Make a small, testable change behind a flag.", "Write unit tests for the new behavior.", "Benchmark against the unmodified engine on two workloads.", "Profile to check overhead; report results honestly, including no-gain outcomes."]
question: "Give an example of a well-scoped modification for this project and how you would evaluate it."
answer: "Example: change the scheduler to prioritize requests with the most prefix cache hits (longest-prefix-first) within a fairness bound. Evaluate on a shared-prefix workload and a random workload, measuring hit rate, TTFT p50 and p95, and fairness (max wait time), comparing against the unmodified engine with identical settings, plus a profile to confirm no added scheduler overhead."
sources: [{"title": "vLLM GitHub repository", "url": "https://github.com/vllm-project/vllm", "note": "Code to modify."}, {"title": "SGLang paper", "url": "https://arxiv.org/abs/2312.07104", "note": "Cache-aware scheduling idea."}]
tags: ["project"]
tools: ["Project"]
---

## Example ideas

| Component | Change |
| --- | --- |
| Scheduler | priority by prefix hit, or a shortest-job-first approximation |
| KV eviction | frequency-aware instead of pure LRU |
| Sampling | a faster top-p kernel |
| Metrics | a new metric for per-tenant queue time |
| Router | cache-aware routing across several replicas |

## Process

1. **Fork** at a pinned version.
2. **Implement behind a config flag,** so the default behavior is unchanged.
3. **Write unit tests** for the new behavior and for the flag being off.
4. **Benchmark** on a workload that should benefit and on one that should not.
5. **Profile** scheduler time or kernel time, to make sure the change does not cost more than it saves.
6. **Write up** the change, the results and the trade-offs.

<div class="ex">
<div class="lab">Worked example</div>

**frequency-aware eviction**

The change: keep a hit counter per cached block, and evict the block with the lowest count among the oldest 10% instead of the single oldest block.

Workload that should benefit: 30% of requests share one of five long system prompts, mixed with a stream of one-off prompts. Expect the prefix hit rate to rise, and TTFT p50 to fall.

Workload that should not: all unique prompts. Expect no change in hit rate. Check that the extra bookkeeping adds less than 1% to scheduler time.

</div>

## Deep dive: why plain LRU fails under scans

LRU keeps the ⟦m1135⟧ most recently used blocks. A burst of ⟦m1136⟧ distinct blocks used once each (a batch of unrelated long documents) evicts everything, including a system prompt used ten times a second. After the burst, that prompt misses until it is recomputed.

Frequency-aware policies separate "seen once" from "seen often". A simple two-queue design admits new blocks into a probation queue of size ⟦m1137⟧, and promotes a block to the protected queue only on its second hit:

⟦M1134⟧

<div class="ex">
<div class="lab">Worked example</div>

**a nightly batch through a shared cache**

Cache: 50,000 blocks. Hot prefixes: 2,000 blocks with 95% of hits. A batch job streams 400,000 one-off blocks.

LRU: the hot set is flushed within the first 50,000 blocks of the batch. The hit rate for interactive traffic drops from 80% to near 0 for the duration of the job.

Two-queue with 10,000 probation blocks: the job churns only the probation queue. Interactive hit rate stays at about 80%.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

a small, flagged, tested change with a fair benchmark teaches more about an engine than reading alone.

</div>
