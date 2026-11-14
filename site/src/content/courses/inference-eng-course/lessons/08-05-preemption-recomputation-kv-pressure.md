---
id: "08.05"
course: "inference-eng-course"
module: "08"
slug: "preemption-recomputation-kv-pressure"
title: "Preemption, Recomputation & KV Pressure"
keyIdeas: ["KV pressure arises because outputs grow unpredictably after admission.", "When blocks run out, the scheduler preempts sequences to free memory.", "Recompute: discard KV, redo prefill later. Swap: move KV to CPU and back.", "Preemption keeps the server alive but hurts tail latency of victims.", "Avoid it by admitting conservatively (watermarks) and sizing for realistic output lengths."]
question: "When is recomputation a better preemption strategy than swapping to CPU, and when is swapping better?"
answer: "Recomputation is better when the preempted sequence is short or the GPU has spare compute, since redoing a short prefill is fast and needs no PCIe transfer or CPU memory. Swapping is better for long sequences, where recomputing a long prefix costs much more than copying the KV over PCIe, provided there is CPU memory and PCIe bandwidth to spare."
sources: [{"title": "Efficient Memory Management for LLM Serving with PagedAttention", "url": "https://arxiv.org/abs/2309.06180", "note": "Swapping vs recomputation analysis."}, {"title": "vLLM: Optimization and tuning", "url": "https://docs.vllm.ai/en/latest/configuration/optimization.html", "note": "Preemption behaviour and settings."}]
tags: ["paging", "scheduling"]
---

## Why memory pressure happens

The scheduler admits a request when its prompt fits in the pool. It cannot know how long the output will be. After admission, every running sequence keeps growing by one block every 16 tokens. So the pool can fill even when nothing new is admitted, simply because the running sequences are getting longer.

<div class="ex">
<div class="lab">Worked example</div>

**how a full pool sneaks up**

The pool has 26,000 blocks. 200 running sequences each hold 100 blocks (1,600 tokens), using 20,000 blocks.

Each sequence needs a new block every 16 steps. At 15 ms per step, the pool loses 200 blocks every 240 ms.

The remaining 6,000 blocks are gone in about 7 seconds, even if no new request arrives.

</div>

## What the engine does when the pool is full

~~~dg-flow Handling a full KV pool, in order of preference.
Need a block, none free -> Evict cached blocks -> Still none -> *Preempt a running sequence -> Requeue it
~~~

1. **Evict cached blocks.** Blocks kept for prefix caching with a reference count of zero are the cheapest to give up.
2. **Preempt a running sequence.** If no cached blocks remain, pick a victim, usually the most recently admitted sequence, free its blocks and put it back in the waiting queue.
3. **Bring the victim back later.** When memory frees up, either recompute its cache or swap it back in.

## Recompute or swap

| | Recompute | Swap |
| --- | --- | --- |
| Extra memory | none | CPU RAM to hold the KV |
| Cost to resume | prefill again over prompt plus generated tokens | PCIe copy back to the GPU |
| Cost to preempt | free the blocks | PCIe copy out to the CPU |
| Best for | short contexts | long contexts |
| Complexity | low | higher: CPU block pool, transfer scheduling |

A PCIe Gen5 x16 link sustains roughly 50 to 60 GB/s in each direction in practice, and Gen4 about half that.

<div class="ex">
<div class="lab">Worked example</div>

**comparing the costs**

A victim with 1,000 tokens of context holds 128 MB of KV for Llama 3 8B. Swapping it out and back over Gen5 takes about 2 × 2.5 ms = 5 ms. Recomputing 1,000 tokens of prefill takes about 25 to 30 ms on an H100. Both are small.

A victim with 16,000 tokens holds 2 GB. Swapping costs about 2 × 40 ms = 80 ms. Recomputing 16,000 tokens takes about 0.5 s, and it competes with other prefills for compute.

For long contexts, swapping is much cheaper. Recent vLLM versions default to recompute for simplicity; check what your engine does.

</div>

## The hidden cost

Preemption keeps the server alive and throughput high, but the preempted requests pay heavily: they queue again and redo or reload their context. Under sustained pressure, the same request can be preempted several times. Its latency can become many times the median while the average barely moves. This is why preemption counts belong on every dashboard.

## Avoiding preemption

- **A free-block watermark.** Admit new requests only if, after allocating their prompt, a reserve of free blocks remains for growth. A watermark of 1 to 5% of the pool is common.
- **A realistic max_num_seqs.** If your typical sequence uses 2,000 tokens and the pool holds 400,000, a cap above about 180 invites preemption.
- **More capacity.** FP8 KV cache doubles the tokens per byte. Quantized weights free memory for the pool. Tensor parallelism spreads the cache across GPUs.
- **Sensible output limits.** A default max_tokens of 1,000 instead of the full context stops runaway generations from consuming the pool.
- **Admission control.** At the gateway or in the engine, reject or queue requests when the pool is nearly full, so the engine does not admit work it cannot finish.

~~~dg-raw
%%widget kv Push the batch until the pool is overcommitted. With FP8 KV in mind, imagine the same bar at half the height.
model = Llama 3 8B
gpu = L40S (48 GB)
t = 12
b = 48
%%
~~~

## Deep dive: how fast a pool fills

Each decode step adds one token to every running sequence. With ⟦m574⟧ running and ⟦m575⟧ free token slots, the pool is full after about

⟦M573⟧

if nothing finishes. Finishing sequences return blocks, so the real balance is admissions plus growth against completions.

<div class="ex">
<div class="lab">Worked example</div>

**ten seconds from comfortable to preempting**

Pool 400,000 tokens. 200 sequences at 1,500 tokens average use 300,000. Free: 100,000.

Growth: 200 tokens per step. At 20 ms per step, the pool fills in ⟦m576⟧ steps, 10 s, unless at least as many tokens finish as are added.

If a burst of long answers arrives together, completions stop for a while, and the pool hits 100% with no warning in the KV usage graph beforehand.

</div>

~~~dg-raw
%%chart KV pool usage during a burst of long answers (illustrative). Each drop is a preemption.
x = 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20
xlab = Seconds
ylab = KV pool used (%)
Pool usage: 75, 79, 83, 87, 91, 95, 99, 96, 99, 97, 99
Preemption threshold*: 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100
%%
~~~

The sawtooth at the top is the engine evicting a sequence, admitting the queue, and filling again. Each tooth is one victim's repeated prefill (07.05 has the recompute and swap costs).

<div class="take">
<div class="lab">Key takeaway</div>

running sequences keep growing, so the KV pool can fill without new admissions. Engines evict cached blocks first and preempt running sequences last. Recompute short victims and swap long ones, and tune admission and capacity so preemption is rare.

</div>
