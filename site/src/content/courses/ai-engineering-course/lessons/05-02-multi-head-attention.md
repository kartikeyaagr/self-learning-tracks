---
id: "05.02"
course: "ai-engineering-course"
module: "05"
slug: "multi-head-attention"
title: "Multi-head attention"
keyIdeas: ["Instead of one big attention, split Q, K and V into h smaller heads that attend independently.", "Each head has dimension d_model / h (4096 / 32 = 128). Total compute is about the same as one full-size head.", "Different heads learn different patterns: previous token, syntax, coreference, copying.", "Outputs of all heads are concatenated back to d_model and passed through an output projection W_O.", "Multi-query (MQA) and grouped-query attention (GQA) share K/V across heads to shrink the KV cache with little quality loss.", "Tensor shapes: (batch, seq, d_model) -> reshape to (batch, heads, seq, head_dim) -> attention -> back."]
question: "A model has d_model = 4096 and 32 heads. What is each head's dimension, and how do multi-head outputs get back to size 4096?"
answer: "Head dimension = 4096 / 32 = 128. Q, K and V are reshaped into 32 heads of 128 dims; each head computes attention independently; the 32 outputs (each seq x 128) are concatenated back to seq x 4096 and multiplied by the output projection W_O (4096 x 4096) to mix information across heads."
sources: [{"title": "Jay Alammar: The Illustrated Transformer", "url": "https://jalammar.github.io/illustrated-transformer/", "note": "The multi-head section shows heads attending to different words."}, {"title": "Vaswani et al.: Attention Is All You Need (2017)", "url": "https://arxiv.org/abs/1706.03762", "note": "Section 3.2.2: multi-head attention."}, {"title": "Shazeer: Fast Transformer Decoding, multi-query attention (2019)", "url": "https://arxiv.org/abs/1911.02150", "note": "One shared K/V head to speed up decoding."}]
---

## Why more than one head

A single attention head computes one set of weights per token: one way of deciding what matters. But a token often needs several kinds of information at once: the previous word, the subject of the verb, the noun a pronoun refers to. Multi-head attention runs several smaller attentions in parallel, each with its own Q, K and V projections, so each can specialise.

## How it's computed

With d_model = 4,096 and h = 32 heads, each head works in 128 dimensions.

~~~dg-flow Multi-head attention for one layer.
X (seq x 4096) -(W_Q, W_K, W_V)-> Q, K, V -(split into 32 heads)-> 32 x (seq x 128) -(attention per head)-> *32 head outputs -(concat)-> seq x 4096 -(W_O)-> Output
~~~

```
x: (batch, seq, 4096)
q = x @ W_Q            -> (batch, seq, 4096)
q = q.view(batch, seq, 32, 128).transpose(1, 2)  -> (batch, 32, seq, 128)
# same for k, v
out = softmax(q @ k.T / sqrt(128)) @ v            -> (batch, 32, seq, 128)
out = out.transpose(1, 2).reshape(batch, seq, 4096)
out = out @ W_O
```

Because the heads split the dimension rather than add to it, total compute is about the same as one full-width head. You get h different attention patterns for free.

## What heads learn

Alammar's Illustrated Transformer shows two heads for the word "it": one focused on "animal", another on "tired". Interpretability research has found heads that track the previous token, heads that match syntax, and induction heads that copy patterns (lesson 04.05). Many heads are redundant; some can be pruned with little loss.

## Sharing keys and values

For decoding speed, the KV cache matters more than compute. Variants:

| Variant | Query heads | K/V heads | KV cache vs MHA |
| --- | --- | --- | --- |
| Multi-head (MHA) | h | h | 1x |
| Grouped-query (GQA) | h | g (e.g. 8) | g / h (e.g. 1/4) |
| Multi-query (MQA) | h | 1 | 1 / h |

Shazeer's multi-query attention paper showed one shared K/V head makes decoding much faster. GQA (Llama 2 70B, Llama 3, Mistral) is the middle ground that keeps quality close to MHA.

## Heads, groups and memory

Multi-head attention splits d_model into h heads that each attend independently. Variants differ in how many key/value heads they keep. Pick a variant and see the parameter count and KV cache change.

~~~dg-raw
%%widget heads Multi-head, grouped-query and multi-query attention: weights and KV cache per token.
~~~

| Variant | Query heads | KV heads | KV cache | Used in |
| --- | --- | --- | --- | --- |
| MHA | h | h | 1× | Original Transformer, GPT-2, Llama 1 |
| GQA | h | g (for example 8) | g / h | Llama 2 70B, Llama 3, Mistral, most current open models |
| MQA | h | 1 | 1 / h | PaLM, Falcon |

GQA (Ainslie et al., 2023) is the compromise that won: with 8 KV groups for 32 or 64 query heads, quality is close to full MHA and the KV cache is 4 to 8× smaller. MQA saves more memory but tends to lose a bit more quality and can be less stable to train.

## The reshape, step by step

For d_model = 4,096 and 32 heads:

1. x has shape [batch, seq, 4,096].
2. Q = x · W_q gives [batch, seq, 4,096]. Reshape to [batch, seq, 32, 128], then transpose to [batch, 32, seq, 128].
3. Each of the 32 heads computes its own [seq, seq] attention matrix and a [seq, 128] output.
4. Transpose back and reshape to [batch, seq, 4,096], then multiply by W_o.

No extra parameters come from having more heads: 32 heads of 128 and 1 head of 4,096 use the same W_q, W_k, W_v and W_o. Heads only change how the same projections are split, so the model can attend to several things at once.

## Do heads specialise?

Some do, measurably. Researchers have found heads that attend to the previous token, heads that copy from an earlier occurrence of the current token ("induction heads", Olsson et al., 2022) and heads that track syntax. Many heads can also be pruned with little loss, which suggests a lot of redundancy. Treat "head 7 does grammar" stories as partial descriptions, not a design you can rely on.

<div class="take">
<div class="lab">Key takeaway</div>

multi-head attention splits d_model into h heads of size d_model/h, runs attention in each, concatenates, and projects with W_O. Sharing K/V across heads (GQA, MQA) keeps the benefit while shrinking the cache.

</div>
