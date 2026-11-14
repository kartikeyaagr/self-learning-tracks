---
id: "27.02"
course: "inference-eng-course"
module: "27"
slug: "follow-one-request-end-to-end"
title: "Follow One Request End-to-End"
keyIdeas: ["Trace one request from HTTP arrival to final token.", "Record each component, function and data structure it passes through.", "Note process boundaries and how data crosses them (IPC, shared memory).", "Time each stage with logs or a profiler.", "Produce a sequence diagram you can reuse to debug."]
question: "What artifacts should you produce from tracing one request through the engine?"
answer: "A sequence diagram of the request across processes and components, with function names; the data structures it is converted into at each stage (request object, scheduled batch, model input tensors, outputs); timestamps of each stage from logs or a profile; and notes on where queueing, allocation, execution and detokenization happen."
sources: [{"title": "vLLM: Architecture overview", "url": "https://docs.vllm.ai/en/latest/design/arch_overview.html", "note": "Component map."}, {"title": "vLLM GitHub repository", "url": "https://github.com/vllm-project/vllm", "note": "Source."}]
tags: ["vllm", "internals"]
---

## Steps

1. **Add log lines or breakpoints** at each of these points: request received, tokenized, added to the engine, scheduled for the first time, first forward pass, first token sampled, detokenized, sent to the client, finished.
2. **Run one request** and collect the timestamps.
3. **Write down the data structure** at each point: what type it is, what fields it has, which process holds it.

## The path

~~~dg-seq One request through a modern engine.
API > Engine core : new request
Engine core > Scheduler : add to waiting
Scheduler > Model runner : scheduled batch
Model runner > Engine core : sampled tokens
~~~

Between the scheduler and the model runner, the scheduler also asks the KV manager for blocks. After the engine core receives sampled tokens, the outputs go back to the API process, which detokenizes and streams them.

<div class="ex">
<div class="lab">Worked example</div>

**a timestamp trace for one request**

Request received at t = 0. Tokenized at 1.2 ms. Added to the engine at 1.5 ms.

Scheduled at 2.1 ms, in the next step after it arrived. Forward pass for 1,500 prompt tokens from 2.4 to 41 ms.

First token sampled at 41.3 ms, detokenized and sent at 42 ms.

TTFT is 42 ms, of which 39 ms is the forward pass. Everything else adds about 3 ms. If your measured TTFT is much higher, the extra time is queueing, not this path.

</div>

## Questions to answer

- Which process runs tokenization, and which runs the scheduler?
- How are outputs sent back across processes, and how are they batched?
- Where is the first token's time spent?

## Deep dive: where one request's time goes across processes

A serving engine splits work across processes so that CPU work overlaps GPU work. One streamed request passes through each of them:

~~~dg-raw
%%timeline One short request through a multi-process engine (ms, illustrative, batch of one).
unit = ms
tick = 5
API server | 0-1.5 parse and tokenize c4 ; 36-37 detokenize c4 ; 44-45 detokenize c4
Engine core | 1.5-2.5 queue and schedule c2 ; 33-34 schedule c2 ; 41-42 schedule c2
GPU worker | 2.5-33 prefill c1 ; 34-41 decode c3 ; 42-49 decode c3
Client | 37-38 first token go ; 45-46 token go
%%
~~~

⟦M1133⟧

Each hop is small, but at batch 1 they add a few milliseconds that the GPU timeline alone does not show. Timestamps logged at each boundary (as in the example above) find them.

<div class="take">
<div class="lab">Key takeaway</div>

tracing one request with timestamps and data structures turns an unfamiliar codebase into a map you can debug with.

</div>
