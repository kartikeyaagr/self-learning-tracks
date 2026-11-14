---
id: "05.03"
course: "ai-engineering-course"
module: "05"
slug: "trace-the-shapes-through-a-transformer"
title: "Trace the shapes through a transformer"
keyIdeas: ["Input token ids: (B, T).", "Embedding lookup: (B, T, d_model).", "Q, K, V after projection and head split: (B, h, T, d_head), with d_head = d_model / h.", "Scores Q K^T: (B, h, T, T). Softmax and multiply by V: (B, h, T, d_head).", "Merge heads and project: (B, T, d_model). MLP expands to (B, T, d_ff) and back. Shape is unchanged between blocks.", "LM head: (B, T, vocab). At inference only the last position's row is sampled."]
question: "For batch 2, sequence 10, d_model 512, 8 heads, vocab 32,000: give the tensor shape after embedding, after splitting into heads, the attention score matrix, and the final logits."
answer: "After embedding: (2, 10, 512). After splitting into heads: (2, 8, 10, 64) for each of Q, K, V (d_head = 512/8 = 64). Attention scores QK^T: (2, 8, 10, 10). Final logits: (2, 10, 32000)."
sources: [{"title": "Andrej Karpathy: nanoGPT model.py", "url": "https://github.com/karpathy/nanoGPT/blob/master/model.py", "note": "Read the shapes in real code."}, {"title": "Jay Alammar: The Illustrated GPT-2", "url": "https://jalammar.github.io/illustrated-gpt2/", "note": "Matrix shapes at each stage, with pictures."}, {"title": "Hugging Face: Llama model code", "url": "https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py", "note": "A production decoder with GQA and RoPE."}]
tags: ["Exercise"]
---

## Why trace shapes

Most bugs in model code are shape bugs, and most performance reasoning starts from shapes: which matrix is T x T, which grows with batch, which grows with vocabulary. This exercise follows one batch through a decoder.

We use: batch B = 2, sequence T = 10, d_model = 512, heads h = 8 (so d_head = 64), MLP size d_ff = 2,048, vocabulary V = 32,000.

## The trace

| Stage | Shape | Example |
| --- | --- | --- |
| Token ids | (B, T) | (2, 10) |
| Token embedding | (B, T, d_model) | (2, 10, 512) |
| Q, K, V projections | (B, T, d_model) each | (2, 10, 512) |
| Split into heads | (B, h, T, d_head) | (2, 8, 10, 64) |
| Scores Q K^T | (B, h, T, T) | (2, 8, 10, 10) |
| Softmax(scores) x V | (B, h, T, d_head) | (2, 8, 10, 64) |
| Merge heads + W_O | (B, T, d_model) | (2, 10, 512) |
| MLP up | (B, T, d_ff) | (2, 10, 2048) |
| MLP down | (B, T, d_model) | (2, 10, 512) |
| After N blocks + final norm | (B, T, d_model) | (2, 10, 512) |
| LM head | (B, T, V) | (2, 10, 32000) |

~~~dg-flow The shape stays (B, T, d_model) between blocks; only attention scores and logits are different.
(B, T) -(embed)-> (B, T, 512) -(heads)-> (B, 8, T, 64) -(QK^T)-> *(B, 8, T, T) -(x V, merge)-> (B, T, 512) -(LM head)-> (B, T, 32000)
~~~

## What the shapes tell you

- **Every block keeps (B, T, d_model).** That's what makes blocks stackable and residual adds possible.
- **The score matrix is (B, h, T, T).** It's the only tensor quadratic in T. At T = 100,000 one head's scores alone are 10^10 entries, which is why FlashAttention never materialises it.
- **Logits are (B, T, V).** With a 128k vocabulary, logits for a long sequence are huge. In training they are often computed in chunks; at inference only the last position is needed.
- **Parameter count** comes from weight shapes, not activations: W_Q, W_K, W_V, W_O are each 512 x 512; MLP weights are 512 x 2,048 and 2,048 x 512.

## With GQA and a KV cache

In a Llama-style model with 2 KV heads instead of 8, K and V become (B, 2, T, 64) and are repeated across 4 query heads each. During decode, Q is (B, h, 1, 64) while the cached K and V are (B, kv_heads, T_past, 64), so scores are (B, h, 1, T_past): linear in context, not quadratic.

## Trace a batch through the model

Set the batch size, sequence length and model width, and follow the tensor shape through every stage of a forward pass.

~~~dg-raw
%%widget shapes Tensor shapes through embedding, attention, MLP and the output head.
~~~

Reading the trace, three facts stand out:

- **The residual stream never changes shape.** [B, T, d_model] goes in to every block and comes out of every block. That is what makes blocks stackable.
- **The attention score tensor is the only one that grows with T².** [B, heads, T, T] is the memory problem FlashAttention (06.03) avoids by never storing it.
- **The output head is huge at the end.** [B, T, vocab] logits with a 128k vocabulary is 128,000 numbers per position. During training this tensor alone can take more memory than the activations of a whole layer; libraries compute the loss in chunks to avoid materialising it.

## Shapes during generation

In decode with a KV cache, the new token has T = 1:

```
x          [B, 1, d_model]
Q          [B, h, 1, d_head]
K, V cache [B, g, T_past + 1, d_head]   (g = KV heads, shared by h / g query heads)
scores     [B, h, 1, T_past + 1]
output     [B, 1, d_model] -> logits [B, 1, vocab]
```

The work per step is now a thin matrix-vector product rather than a matrix-matrix product. That is why decode is bound by memory bandwidth (reading weights and cache) rather than by compute, and why batching many users together is so effective.

<div class="ex">
<div class="lab">Worked example</div>

**a shape bug caught early**

You implement GQA and forget to repeat the 8 KV heads to match 32 query heads. The matmul of [B, 32, T, 128] by [B, 8, 128, T] fails with a shape error, or worse, broadcasts silently in a library that allows it.

Writing the expected shape next to every line, as in the trace above, catches this before any training run.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

(B, T) to (B, T, d) to (B, h, T, d_head) to (B, h, T, T) and back to (B, T, d), then (B, T, vocab). Knowing where T x T and vocab appear tells you where memory goes.

</div>
