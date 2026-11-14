---
id: "07.05"
course: "inference-eng-course"
module: "07"
slug: "queueing-fairness-cancellation-preemption"
title: "Queueing, Fairness, Cancellation & Preemption"
keyIdeas: ["Default policy is first come, first served; simple but one heavy user can hurt others.", "Fairness policies (per-tenant quotas, virtual token counters) protect light users.", "Priorities let interactive traffic jump ahead of batch traffic.", "Cancellation must free KV right away when clients disconnect.", "Preemption evicts running sequences when KV runs out: recompute later or swap to CPU."]
question: "A client disconnects mid-stream. What should the engine do and why does it matter for capacity?"
answer: "The engine should detect the disconnect, abort the request, stop generating for it and free its KV blocks immediately. Otherwise it keeps spending decode steps and KV memory on tokens nobody will read, reducing capacity for live users. In busy chat products a meaningful share of requests are abandoned, so this matters."
sources: [{"title": "vLLM: Optimization and tuning", "url": "https://docs.vllm.ai/en/latest/configuration/optimization.html", "note": "Preemption modes."}, {"title": "Fairness in Serving Large Language Models (VTC)", "url": "https://arxiv.org/abs/2401.00588", "note": "Fair scheduling across clients."}]
tags: ["scheduling"]
---

## Who waits when the server is busy

At any moment, an engine under load has requests it cannot yet run. Queue order, cancellation and preemption decide who waits, who gets served, and who pays when memory runs short. These choices shape tail latency and fairness more than any kernel.

## Queue order

- **First come, first served (FCFS).** Requests run in arrival order. Simple, predictable and the default in most engines. Its weakness: one tenant sending a burst of long requests delays everyone behind it.
- **Priority.** Each request carries a priority, such as interactive above batch or paid above free. The scheduler always admits the highest-priority waiting request first. vLLM supports a priority scheduling policy. Without limits, low-priority traffic can starve.
- **Shortest job first.** Admitting short prompts first lowers average TTFT, but long prompts can wait indefinitely, and output lengths are unknown.
- **Fair share.** Track how much service each tenant received, such as tokens processed, and favour the ones who got less. The Virtual Token Counter (VTC) paper adapted this idea to LLM serving, counting input and output tokens with different weights.

<div class="ex">
<div class="lab">Worked example</div>

**why FCFS can be unfair**

Tenant A sends 200 summarization requests with 16k-token prompts at once. Tenant B sends normal chat requests.

Under FCFS, B's requests queue behind 200 × 16k = 3.2 million prompt tokens. At 40,000 prefill tokens per second, that is 80 s of waiting.

Under fair share, the scheduler alternates. B's requests wait a few hundred milliseconds, and A's batch job takes slightly longer.

</div>

## Cancellation is free capacity

Users close browser tabs, agents hit their own timeouts, and clients retry. If the engine keeps generating for a request nobody is reading, it burns decode steps and holds KV blocks for nothing.

Check the whole path:

- The HTTP server must notice the client disconnect.
- The API layer must send an abort to the engine.
- The engine must remove the sequence and free its blocks in the next step.

In some deployments, 5 to 10% of requests are abandoned. Getting cancellation right returns that capacity at no cost.

## Preemption

Sometimes the scheduler cannot continue as it is. Running sequences keep growing, each needing a new KV block every 16 tokens. If the pool is empty, at least one sequence must give up its memory.

~~~dg-flow Preemption under KV pressure.
KV pool full -> Pick a victim, usually the newest -> *Free its blocks -> Later, recompute or swap back
~~~

Engines have two ways to bring the victim back:

- **Recompute.** Discard its KV cache. When it is rescheduled, run prefill again over its prompt plus every token it has generated so far. This needs no extra memory but costs compute.
- **Swap.** Copy its KV cache to CPU memory over PCIe, and copy it back when rescheduled. This saves compute but needs CPU memory and transfer time.

The usual victim is the most recently admitted request. Older requests have done more work, and finishing them frees memory soonest.

## The cost of preemption

A preempted request pays twice: it waits in the queue again, then it redoes or reloads its context. For a request with 3,000 tokens of context, recomputation costs about the same as a fresh 3,000-token prefill. Its total latency can be several times the normal value. Throughput can look fine while a few unlucky users wait much longer.

## Signs of trouble and fixes

- **Preemptions every minute at normal load**: the engine is overcommitting memory. Lower max_num_seqs, or add KV capacity with FP8 KV or more GPUs.
- **Preemptions only during bursts**: acceptable if rare, but consider admission control that rejects early instead of preempting late.
- **Many very long generations**: set sensible default max_tokens. A runaway generation of 30,000 tokens can push dozens of others out.

## Deep dive: recompute or swap?

A preempted sequence holding ⟦m510⟧ tokens can be recomputed (prefill again) or swapped (KV copied to CPU memory and back). Their costs are

⟦M509⟧

The ratio depends only on the model's parameters per byte of KV and on the hardware.

<div class="ex">
<div class="lab">Worked example</div>

**Llama 3 8B on an H100 with PCIe Gen5**

⟦m511⟧: recompute ⟦m512⟧ ms. Swap ⟦m513⟧ ms.

Ratio ⟦m514⟧. Swap is cheaper here.

But swap uses PCIe bandwidth shared with other transfers, and needs pinned CPU memory. Recompute needs neither and can reuse cached prefixes. That is why many engines default to recompute.

</div>

## Deep dive: cancellation in numbers

If a fraction ⟦m515⟧ of streams are abandoned on average halfway through, and the engine does not notice, the wasted decode share is about ⟦m516⟧. With 10% abandoned, 5% of all decode work produces tokens nobody reads. Those sequences also hold KV, which cuts admission for everyone else. Propagating client disconnects to the engine recovers both.

<div class="take">
<div class="lab">Key takeaway</div>

queue order, cancellation and preemption decide who waits and who pays when the server is busy. Fix cancellation first because it is free capacity, choose a fair queue policy for multi-tenant traffic, and treat frequent preemption as a sign of overcommitted memory.

</div>
