---
id: "20.01"
course: "inference-eng-course"
module: "20"
slug: "measure-long-context-scaling"
title: "Measure Long-Context Scaling"
keyIdeas: ["Long context hurts all three: TTFT, TPOT and concurrency.", "Prefill attention grows with length squared; linear parts grow linearly.", "Decode attention reads all KV: TPOT grows with context.", "KV per sequence grows linearly: concurrency shrinks.", "Measure a sweep of context lengths before designing a long-context service."]
question: "Describe how TTFT, TPOT and max concurrency change as context goes from 4k to 128k tokens on one GPU."
answer: "TTFT grows faster than linearly because attention prefill cost grows with length squared, so 128k can take tens of seconds. TPOT grows roughly linearly with context because each decode step reads the whole KV cache, which eventually outweighs weights. Max concurrency falls inversely with context: 32 times longer context means about 32 times fewer sequences fit."
sources: [{"title": "Lost in the Middle", "url": "https://arxiv.org/abs/2307.03172", "note": "Quality behavior over long contexts."}, {"title": "kipply: Transformer Inference Arithmetic", "url": "https://kipp.ly/transformer-inference-arithmetic/", "note": "KV and attention scaling."}]
tags: ["long-context"]
---

## What to sweep

Long context changes three numbers at once, and each changes at a different rate. Measure all three at several lengths: 4k, 16k, 32k, 64k and 128k tokens.

1. **TTFT for a single request.** How long the user waits for the first token.
2. **TPOT for a single request with a long output.** How fast tokens stream once decoding starts.
3. **Maximum concurrency** before the KV cache runs out. How many users one GPU can hold at that length.

## Typical shape

These numbers are illustrative for an 8B model on one H100 with about 50 GB of KV pool.

| Context | TTFT | TPOT | Sequences that fit in KV |
| --- | --- | --- | --- |
| 4k | about 0.2 s | about 7 ms | about 100 |
| 32k | about 2.5 s | about 10 ms | about 12 |
| 128k | 20 s or more | about 20 ms | about 3 |

Each of the three moves differently:

- **TTFT grows faster than linearly**, because attention FLOPs in prefill grow with the square of the prompt length.
- **TPOT grows linearly**, because each decode step reads the whole KV cache once, and the cache grows with context.
- **Concurrency falls in inverse proportion**, because each sequence takes memory in proportion to its length.

~~~dg-raw
%%widget kv Set the context to 128k (17 on the slider) and see how many sequences still fit. Then switch to the model without GQA.
model = Llama 3 8B
gpu = H100 SXM (80 GB)
t = 15
b = 8
%%
~~~

## Where the time goes at long context

- **Prefill:** beyond a few tens of thousands of tokens, attention makes up most of the FLOPs. The next lesson does the arithmetic.
- **Decode:** once context × batch is large, the KV bytes read per step exceed the weight bytes. At 32k context and batch 8 on an 8B model, the KV read is 32k × 128 KB × 8 = 32 GB, twice the 16 GB of weights.

<div class="ex">
<div class="lab">Worked example</div>

**why one 128k request is expensive**

An 8B model's KV is 128 KB per token, so a 128k-token request holds 16 GB of KV.

That is the same memory as about 30 requests at 4k tokens.

If you price by tokens, the 128k request pays for 128k input tokens once. But it occupies the KV of 30 short requests for its whole decode. Long-context traffic often needs its own pricing or its own pool of GPUs.

</div>

## Quality check

Accuracy can fall at long context. Models often use information in the middle of a long prompt less well than information at the start or end. Test retrieval and question answering at the lengths you actually serve, not only at short lengths.

## Deep dive: decode at long context, predicted

At batch 1, each step reads the weights and the whole KV cache:

⟦M962⟧

~~~dg-raw
%%chart Predicted batch-1 TPOT against context, Llama 3 8B on an H100.
x = 1024, 8192, 32768, 65536, 131072
logx = 1
xlab = Context (tokens)
ylab = TPOT (ms)
TPOT: 6.8, 7.1, 8.3, 9.8, 12.9
%%
~~~

At 128k, the KV cache (17 GB) is larger than the weights (16 GB). A single long conversation doubles the bytes per step, and a batch of eight such conversations needs 137 GB of KV: more than one GPU holds.

<div class="take">
<div class="lab">Key takeaway</div>

long context raises TTFT faster than linearly, raises TPOT linearly and cuts concurrency in proportion. Measure all three and plan around KV memory.

</div>
