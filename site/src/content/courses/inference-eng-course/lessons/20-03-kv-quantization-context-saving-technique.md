---
id: "20.03"
course: "inference-eng-course"
module: "20"
slug: "kv-quantization-context-saving-techniques"
title: "KV Quantization & Context-Saving Techniques"
keyIdeas: ["Lossless or near lossless: FP8 KV, architectures with GQA or MLA.", "Architectural limits: sliding-window or local attention layers.", "Lossy runtime methods: evict or compress tokens by importance.", "MLA stores a compressed latent per token instead of full K and V.", "Evaluate on tasks needing details from far back in the context."]
question: "Compare FP8 KV, sliding-window attention and KV eviction (like H2O or StreamingLLM) as ways to fit long contexts."
answer: "FP8 KV halves memory with little quality loss and keeps all tokens. Sliding-window attention (a model architecture choice) keeps only the last W tokens' KV for some or all layers, bounding memory, but only works if the model was trained that way. KV eviction methods drop tokens judged unimportant (keeping attention sinks and heavy hitters) to fit a budget; they work with existing models but are lossy and can miss details needed later."
sources: [{"title": "StreamingLLM (attention sinks)", "url": "https://arxiv.org/abs/2309.17453", "note": "Keeping initial tokens for stable streaming."}, {"title": "H2O: Heavy-Hitter Oracle", "url": "https://arxiv.org/abs/2306.14048", "note": "Evicting low-importance KV."}, {"title": "DeepSeek-V2 (MLA)", "url": "https://arxiv.org/abs/2405.04434", "note": "Multi-head latent attention compresses KV."}]
tags: ["long-context", "kv-cache"]
---

## Techniques

| Technique | Memory saving | Quality risk | Needs |
| --- | --- | --- | --- |
| FP8 KV cache | 2 times | small | kernel support |
| INT4 or 2-bit KV | 4 to 8 times | moderate | special kernels |
| GQA or MQA | 4 to 32 times compared with full multi-head | built into model | model choice |
| MLA | large | built into model | model choice |
| Sliding-window layers | bounded per layer | built into model | model choice |
| Eviction (H2O, attention sinks) | configurable | can lose details | engine support |

## Lossless first

The first three rows of savings that come from the model or from FP8 are close to free in quality. They should be in place before you try anything lossy.

<div class="ex">
<div class="lab">Worked example</div>

**KV per token for three designs**

Llama 2 7B (no GQA): 2 × 32 layers × 32 heads × 128 × 2 bytes = 512 KB per token.

Llama 3 8B (GQA with 8 KV heads): 2 × 32 × 8 × 128 × 2 = 128 KB per token. Four times smaller.

The same model with FP8 KV: 64 KB per token. Eight times smaller than the first model. At 32k context, that is 16 GB, 4 GB and 2 GB per sequence.

</div>

## MLA

Multi-head latent attention, used in DeepSeek models, stores one small latent vector per token instead of separate K and V for each head.

- At attention time, the per-head K and V are rebuilt from the latent, or the projection is folded into the query so they never need to be rebuilt.
- DeepSeek-V3 stores about 576 values per token per layer, around 70 KB per token in BF16 across 61 layers. A GQA model of similar size would need several times more.
- The cost is more compute in attention, which is a good trade in memory-bound decode.

## Sliding-window layers

Some models use a fixed window (for example 4k tokens) in most layers and full attention only in a few. The windowed layers' KV stops growing past the window, so total KV grows much more slowly with context.

## Eviction

Eviction methods keep:

- the first few tokens (attention sinks, which many heads attend to regardless of content),
- the most recent window,
- tokens that have received high attention so far.

They drop the rest. This works well for streaming chat, where old details rarely matter. It is risky for retrieval over long documents, where the one dropped sentence may be the answer.

## Deep dive: KV per token for four designs

⟦M965⟧

For a sliding window ⟦m967⟧ on some layers, KV per sequence stops growing on those layers:

⟦M966⟧

<div class="ex">
<div class="lab">Worked example</div>

**MLA in DeepSeek-V3**

61 layers, a compressed latent of ⟦m968⟧ plus a 64-dimensional RoPE key, in BF16: ⟦m969⟧ bytes, about 69 KB per token.

Full multi-head attention with its 128 heads of dimension 128 would store ⟦m970⟧ MB per token, 57 times more.

</div>

~~~dg-raw
%%chart KV per sequence against context: all-global GQA versus 6 global and 26 sliding-window layers (W = 1,024), with Llama 3 8B's per-layer KV size.
x = 1024, 8192, 32768, 131072
logx = 1
logy = 1
xlab = Context (tokens)
ylab = GB per sequence
All layers global: 0.134, 1.07, 4.29, 17.2
Mostly sliding window: 0.134, 0.31, 0.91, 3.33
%%
~~~

At 128k the hybrid design needs a fifth of the KV. The price is that only 6 layers can look further back than 1,024 tokens, which has to be checked against long-context evaluations.

<div class="take">
<div class="lab">Key takeaway</div>

use lossless savings first: FP8 KV and efficient architectures such as GQA, MLA and sliding windows. Lossy eviction can fit more, but test it on tasks that need distant details.

</div>
