---
id: "28.04"
course: "inference-eng-course"
module: "28"
slug: "add-continuous-batching-scheduling"
title: "Add Continuous Batching & Scheduling"
keyIdeas: ["Add a waiting queue and a running set.", "Each step: retire finished, admit new within budget, schedule mixed prefill + decode.", "Support chunked prefill via a token budget.", "Handle cancellation and stop conditions per request.", "Compare throughput and latency with static batching on the same workload."]
question: "What changes in your engine loop when you move from static to continuous batching?"
answer: "The loop no longer processes a fixed batch to completion. Each step it removes finished sequences, admits waiting requests if there is KV space and token budget, schedules their prefill (whole or chunked) alongside current decodes, runs one forward pass over the mixed batch, and emits tokens. Requests carry state (computed tokens, status) so they can join and leave at any step."
sources: [{"title": "Anyscale: How continuous batching enables 23x throughput", "url": "https://www.anyscale.com/blog/continuous-batching-llm-inference", "note": "Concept and gains."}, {"title": "Sarathi-Serve", "url": "https://arxiv.org/abs/2403.02310", "note": "Chunked prefill in the loop."}]
tags: ["mini-engine", "scheduling"]
---

## The loop

```
while True:
    retire_finished(); handle_cancellations()
    batch = scheduler.schedule()           # decodes plus prefill chunks within the budget
    if not batch:
        wait_for_requests(); continue
    logits = runner.forward(*batch.tensors())
    tokens = sampler(logits, batch.params)
    batch.update(tokens)                   # append, stream, mark finished
```

Each step, the scheduler decides afresh which requests run. A finished request leaves at once and a new one can join in the very next step.

## Mixed batch tensors

Flatten all scheduled tokens into one sequence. The metadata says, for each request:

- where its tokens start in the flat batch,
- its total context length,
- whether this step finishes its prompt, so a token should be sampled for it.

~~~dg-raw
%%widget cb Run both schedulers on the same random lengths. Count the idle slots in static batching.
%%
~~~

## Evaluation

Run your benchmark harness from Module 04 against the static and continuous versions, at concurrency 1 to 64.

| Metric | Static | Continuous |
| --- | --- | --- |
| Throughput at concurrency 32 | | |
| TTFT p95 at concurrency 32 | | |

## What to expect

- Throughput at concurrency 32 rises, often by 2 to 3 times, because slots no longer sit idle.
- TTFT p95 falls sharply, because new requests no longer wait for a whole batch to finish.
- At concurrency 1, the two versions are the same.

<div class="take">
<div class="lab">Key takeaway</div>

continuous batching is a scheduler plus per-request state in the engine loop. It should raise throughput and cut TTFT immediately.

</div>
