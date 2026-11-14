---
id: "27.03"
course: "inference-eng-course"
module: "27"
slug: "follow-the-scheduler-request-state"
title: "Follow the Scheduler & Request State"
keyIdeas: ["Find the scheduler's main schedule function and its output structure.", "Request state: computed tokens, target tokens, status, blocks.", "Order: running requests first, then waiting; budget and KV checks.", "Preemption path when blocks run out.", "Write a small test that exercises budget limits and preemption."]
question: "What state does the scheduler keep per request, and how does it decide how many tokens a request gets this step?"
answer: "Per request: status (waiting, running, preempted, finished), number of tokens computed so far, number of prompt and output tokens, assigned KV blocks, and sampling parameters. Each step it gives running requests their decode token (or next prefill chunk), then fills remaining token budget with waiting requests' prompt tokens, limited by free KV blocks and max sequences; it preempts if decode cannot get blocks."
sources: [{"title": "vLLM GitHub repository", "url": "https://github.com/vllm-project/vllm", "note": "Scheduler source."}, {"title": "Sarathi-Serve", "url": "https://arxiv.org/abs/2403.02310", "note": "Token-budget scheduling."}]
tags: ["scheduler", "internals"]
---

## The unified view

Modern schedulers treat prefill and decode the same way. Each request has num_computed_tokens and a target number of tokens. Each step, the scheduler gives it min(remaining, budget) tokens.

- A new request with a 3,000-token prompt has 3,000 tokens remaining.
- A decoding request has 1 token remaining each step.
- A request whose prompt was partly prefilled last step has the rest remaining.

There is no separate "prefill phase" in the code. Chunked prefill falls out naturally from the token budget.

## Pseudocode of one step

```
budget = max_num_batched_tokens
for r in running:
    n = min(r.remaining_tokens, budget)
    if not kv.can_allocate(r, n):
        preempt(lowest_priority_running)
        continue
    schedule(r, n); budget -= n
for r in waiting:
    if budget == 0 or len(running) == max_num_seqs:
        break
    hit = kv.prefix_hit(r)
    n = min(r.remaining_tokens - hit, budget)
    if kv.can_allocate(r, n):
        admit(r, n); budget -= n
```

<div class="ex">
<div class="lab">Worked example</div>

**one step with a budget of 2,048**

Running: 60 decoding requests, 1 token each. That uses 60 tokens, leaving 1,988.

Waiting: request A with a 3,000-token prompt, 1,024 of them cached. It needs 1,976 more, which fits. A is admitted with 1,976 tokens and the budget falls to 12.

Waiting: request B with 500 tokens. Only 12 tokens of budget remain. B gets 12 tokens this step and the rest in later steps, or waits, depending on the engine's policy for partial prefills.

</div>

~~~dg-raw
%%widget cb Compare static and continuous batching on the same random workload. The scheduler loop above is what makes the continuous version possible.
%%
~~~

## What to verify in the code

- How prefix hits reduce the scheduled tokens.
- Where max_num_seqs is enforced.
- Which request is preempted, and what happens to its state: are its blocks freed and recomputed later, or swapped out?

<div class="take">
<div class="lab">Key takeaway</div>

the scheduler is a loop over running and then waiting requests, under a token budget and KV limits. Reading it explains most latency behavior.

</div>
