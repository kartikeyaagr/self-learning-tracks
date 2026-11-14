---
id: "07.01"
course: "inference-eng-course"
module: "07"
slug: "one-by-one-serving-vs-static-batching"
title: "One-by-One Serving vs Static Batching"
keyIdeas: ["One-by-one serving: GPU at batch 1, users queue.", "Static batching: group requests, run until all finish. Better throughput, but padding waste.", "Short requests are held hostage by the longest one in their batch.", "New arrivals wait for the whole batch to end.", "Output lengths vary a lot in LLM traffic, so static batching wastes much of the batch."]
question: "With static batching, a batch has requests needing 20, 50 and 500 output tokens. What happens to the GPU and to the first two requests?"
answer: "All three start together and the batch runs until the 500-token request finishes. The first two finish early but their slots stay idle (padding) for the remaining steps, so GPU work is wasted, and new requests cannot join until the whole batch ends, so they wait. Users of short requests may also wait if results are returned only when the batch completes."
sources: [{"title": "Anyscale: How continuous batching enables 23x throughput", "url": "https://www.anyscale.com/blog/continuous-batching-llm-inference", "note": "Static vs continuous batching explained."}, {"title": "Orca", "url": "https://www.usenix.org/conference/osdi22/presentation/yu", "note": "Motivation for iteration-level scheduling."}]
tags: ["batching"]
---

## One request at a time

The simplest server runs each request alone. The first user gets the best possible latency. Everyone else waits in a queue, and the GPU runs at batch 1 the whole time, using a few percent of what it could do. Lesson 01.05 showed the numbers: ten users arriving together wait between 4 and 40 seconds, while throughput never rises.

## Static batching

The obvious improvement is to group requests. Collect B requests, pad their prompts to the same length, run prefill on all of them together, then decode until every one has finished. This is **static batching**, and it is how most deep learning inference worked before LLMs.

It does share the weight read: a batch of 8 decodes 8 tokens per step for about the cost of one. But LLMs break it in two ways.

## Problem 1: requests finish at different times

Output lengths vary enormously, from 5 tokens for a yes or no answer to thousands for an essay, and you do not know them in advance. In a static batch, a slot whose request finishes early sits empty until the longest request is done.

~~~dg-flow A static batch with very different output lengths.
Start 3 requests -> Short ones finish -(slots idle)-> *Wait for the longest -> Batch ends -> Next batch starts
~~~

| Steps | Active requests | Idle slots |
| --- | --- | --- |
| 1 to 20 | 3 | 0 |
| 21 to 50 | 2 | 1 |
| 51 to 500 | 1 | 2 |

Over the 500 steps, the batch has 1,500 slot-steps, but only 20 × 3 + 30 × 2 + 450 × 1 = 570 do useful work. Utilization is 38%. Most of the time, this "batch of 3" is really a batch of 1.

## Problem 2: new requests wait for the whole batch

A request that arrives one step after a batch started cannot join it. It waits until every request in the batch has finished, then starts in the next batch. If the batch contains one 2,000-token generation, the newcomer's TTFT includes those 2,000 steps, maybe 40 seconds. Short requests are held hostage by long ones they have nothing to do with.

## Padding adds more waste

Prompts in a static batch must be padded to the longest prompt. If one prompt has 3,000 tokens and the others 100, prefill processes 4 × 3,000 = 12,000 positions for 3,300 real tokens. The KV cache must also be allocated for the padded length.

~~~dg-raw
%%widget cb Switch between static and continuous batching on the same requests. Hatched areas are slots doing nothing.
%%
~~~

<div class="ex">
<div class="lab">Worked example</div>

**utilization of static batching in practice**

A chat workload has output lengths with a median of 120 tokens and a long tail to 2,000.

With static batches of 16, the expected longest output in a batch is about 1,100 tokens, while the average is about 200.

Slot utilization is about 200 ÷ 1,100 ≈ 18%. Continuous batching on the same workload keeps slots more than 90% busy.

</div>

## Why images were different

Image classifiers and many vision models produce fixed-size outputs in one forward pass. Every request in a batch does exactly the same work and finishes at the same time, so static batching wastes nothing. LLM generation is a variable number of sequential steps per request, which is why it needed a new kind of batching.

## Deep dive: static batching on a timeline

~~~dg-raw
%%timeline Four requests in one static batch (decode steps). Grey is a slot holding nothing useful; the new request waits for all of it.
unit = steps
tick = 100
Slot 1 | 0-50 decode c3 ; 50-400 idle line
Slot 2 | 0-120 decode c3 ; 120-400 idle line
Slot 3 | 0-400 decode c3
Slot 4 | 0-80 decode c3 ; 80-400 idle line
New request | 0-400 waiting flame ; 400-450 decode go
%%
~~~

Useful work is ⟦m491⟧ slot-steps out of ⟦m492⟧: 41% utilization. The new request's TTFT includes all 400 steps.

## Deep dive: padding waste on the prompt side

Static batches also pad every prompt to the longest one. With prompt lengths ⟦m493⟧:

⟦M490⟧

Prompts of 200, 400, 300 and 2,000 tokens pad to 4 × 2,000 = 8,000 positions to carry 2,900 real ones: 64% waste. The padded positions cost FLOPs in every layer and KV memory for the whole run. The flat token list of continuous batching has no padding at all.

<div class="take">
<div class="lab">Key takeaway</div>

static batching shares weight reads but wastes slots when requests finish at different times, makes new requests wait for the slowest one, and pads prompts. The fix is to let requests join and leave the batch at every step.

</div>
