---
id: "28.02"
course: "inference-eng-course"
module: "28"
slug: "add-kv-cache"
title: "Add KV Cache"
keyIdeas: ["Add per-layer K and V buffers; write during prefill and each decode step.", "Attention reads cached K and V up to current length.", "Separate prefill and decode code paths, or unify with metadata.", "Test against full recompute for exact match under greedy decoding.", "Measure step time vs position before and after."]
question: "What tests prove your KV cache is correct?"
answer: "Greedy generation with and without the cache produces identical tokens for several prompts and lengths; logits at each decode step match a full recompute within tolerance; positions continue correctly after the prompt (RoPE); and step time stays nearly flat as the sequence grows."
sources: [{"title": "Hugging Face: KV cache strategies", "url": "https://huggingface.co/docs/transformers/main/en/kv_cache", "note": "Reference behaviors."}, {"title": "kipply: Transformer Inference Arithmetic", "url": "https://kipp.ly/transformer-inference-arithmetic/", "note": "KV sizing."}]
tags: ["mini-engine", "kv-cache"]
---

## Implementation

- **Allocate** a [max_len, kv_heads, head_dim] buffer for K and one for V, per layer, per sequence. Contiguous buffers are fine for now.
- **Prefill:** compute K and V for all prompt tokens, write them into the buffers, and run causal attention.
- **Decode:** compute K and V for the one new token, write them at position pos, and attend over positions 0 to pos.

Without a cache, each new token recomputes K and V for the entire sequence. With it, each step does work for one token plus a read of the cached keys and values.

## Tests

| Test | Pass condition |
| --- | --- |
| Greedy match | cached and uncached generation give the same tokens for 5 prompts × 200 tokens |
| Logit match | maximum absolute difference below your tolerance at every step |
| RoPE positions | the decode position equals prompt length plus step |
| Speed | step time roughly flat from position 100 to position 2,000 |

<div class="ex">
<div class="lab">Worked example</div>

**what the speed test shows**

Without the cache, an 8B model's step time grows with sequence length: about 20 ms at position 100, about 200 ms at position 2,000, because each step re-runs the whole sequence.

With the cache, step time stays near 20 ms from position 100 to 2,000. It rises only slightly, from reading more KV.

If your cached version still slows down steeply, it is probably recomputing something per step, such as the RoPE table for all positions or a full causal mask.

</div>

~~~dg-raw
%%widget kv The cache you just built has a size you can calculate. See how it grows with context and batch.
model = Llama 3 8B
t = 11
b = 1
%%
~~~

## Deep dive: the speed test, predicted

Without a cache, step ⟦m1142⟧ reruns ⟦m1143⟧ tokens; with it, one token. At batch 1 on a GPU, both are dominated by fixed costs until ⟦m1144⟧ passes the ridge (about 200 tokens for an 8B model), after which the uncached step grows linearly:

⟦M1141⟧

~~~dg-raw
%%chart Predicted step time in a mini engine (8B, H100, 10 ms of Python overhead per step).
x = 100, 250, 500, 1000, 2000
logx = 1
xlab = Tokens in the sequence
ylab = ms per step
No cache: 15.8, 16.3, 22.5, 35, 60
KV cache: 15.8, 15.8, 15.8, 15.8, 15.9
%%
~~~

Python overhead dominates both at short lengths, so the cache shows no gain until a few hundred tokens. If your measured cached line rises with length, look for a concatenation instead of in-place writes (03.04).

<div class="take">
<div class="lab">Key takeaway</div>

a correct KV cache produces exactly the same tokens as recomputation and makes step time flat. Test both.

</div>
