---
id: "26.05"
course: "inference-eng-course"
module: "26"
slug: "reasoning-model-inference"
title: "Reasoning Model Inference"
keyIdeas: ["Reasoning models produce long hidden or visible thinking before answers.", "Workload becomes decode-heavy with very long outputs.", "KV per request grows with thinking length; concurrency drops.", "Levers: thinking budgets, speculative decoding, FP8 KV, decode-optimized pools.", "Measure cost per answered request, not per token."]
question: "A reasoning model generates 5,000 thinking tokens before a 200-token answer. What does this do to capacity and what levers exist?"
answer: "Output tokens grow 25 times, so decode time, KV memory per request (the thinking stays in context) and cost all rise sharply; concurrency per GPU falls and E2E latency becomes tens of seconds. Levers: decode speed optimizations (speculative decoding, quantization), thinking budgets or effort settings, larger KV capacity (FP8 KV, offload), and streaming the thinking or a summary so users see progress."
sources: [{"title": "DeepSeek-R1", "url": "https://arxiv.org/abs/2501.12948", "note": "Reasoning model trained with RL."}, {"title": "vLLM: Reasoning outputs", "url": "https://docs.vllm.ai/en/latest/features/reasoning_outputs.html", "note": "Separating reasoning content in the API."}]
tags: ["reasoning"]
---

## Shape

| Metric | Chat model | Reasoning model |
| --- | --- | --- |
| Output tokens | about 300 | 3,000 to 30,000 |
| End-to-end time | seconds | tens of seconds to minutes |
| KV per request at the end | small | large |
| Bottleneck | mixed | decode |

A reasoning model writes a long chain of thought before its answer. Serving it is mostly a decode problem, and each request occupies KV for a long time.

<div class="ex">
<div class="lab">Worked example</div>

**KV held by one reasoning request**

An 8B model with FP8 KV uses 64 KB per token. A request with a 2,000-token prompt and 20,000 tokens of reasoning ends holding 22,000 × 64 KB ≈ 1.4 GB.

At 15 ms per token, it decodes for about 5 minutes. A GPU with 50 GB of KV room can hold about 35 such requests at once.

The same GPU could hold several hundred chat requests. Capacity per GPU, measured in concurrent users, falls by an order of magnitude.

</div>

## Levers

- **Budget:** cap thinking tokens, or expose effort levels so users choose how long the model thinks.
- **Decode speed:** speculative decoding works well, because reasoning text is often predictable and acceptance rates are high. Quantization helps too.
- **KV:** FP8 KV, more decode GPUs, and disaggregation with decode-heavy pools.
- **Timeouts:** end-to-end timeouts must allow long runs.
- **UX:** stream the reasoning or a progress indicator so TTFT stays meaningful.

~~~dg-raw
%%widget spec Reasoning text often gets high acceptance rates. Set α to 0.8 and find the best draft length.
a = 0.8
k = 5
c = 0.05
%%
~~~

## Economics

Measure **cost per solved task**, not cost per request. A reasoning model may cost 10 times more per request, but if it replaces several retries, or a much larger model, it can still be cheaper per correct answer.

## Deep dive: memory-time of a long reasoning trace

A request with prompt ⟦m1113⟧ that generates ⟦m1114⟧ tokens at TPOT ⟦m1115⟧ holds KV that grows linearly while it runs. The memory-time it consumes is

⟦M1112⟧

which grows with ⟦m1116⟧ once ⟦m1117⟧ exceeds ⟦m1118⟧.

~~~dg-raw
%%chart KV memory-time of one request against reasoning length (p = 1,000, 20 ms TPOT, 128 KB per token).
x = 1000, 4000, 16000, 32000
logx = 1
logy = 1
xlab = Generated tokens
ylab = GB·s of KV
KV memory-time: 3.9, 31, 377, 1426
%%
~~~

A 32k-token reasoning trace uses about 360 times the KV memory-time of a 1k-token answer, for 32 times the tokens. If such requests are priced per output token, they are subsidized by short ones.

<div class="take">
<div class="lab">Key takeaway</div>

reasoning turns serving into long-output decode. Plan for KV growth, budget thinking tokens and optimize decode speed.

</div>
