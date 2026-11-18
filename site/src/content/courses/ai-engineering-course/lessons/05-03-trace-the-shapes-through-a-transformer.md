---
id: "05.03"
course: "ai-engineering-course"
module: "05"
slug: "trace-the-shapes-through-a-transformer"
title: "Trace the shapes through a transformer"
keyIdeas: ["Input token ids: (B, T).", "Embedding lookup: (B, T, d_model).", "Q, K, V after projection and head split: (B, h, T, d_head), with d_head = d_model / h.", "Scores Q K^T: (B, h, T, T). Softmax and multiply by V: (B, h, T, d_head).", "Merge heads and project: (B, T, d_model). MLP expands to (B, T, d_ff) and back. Shape is unchanged between blocks.", "LM head: (B, T, vocab). At inference only the last position's row is sampled."]
question: "For batch 2, sequence 10, d_model 512, 8 heads, vocab 32,000: give the tensor shape after embedding, after splitting into heads, the attention score matrix, and the final logits."
answer: "After embedding: (2, 10, 512), one 512-wide vector per token. After splitting into heads: (2, 8, 10, 64) for each of Q, K and V, since d_head = 512 / 8 = 64; the projections first give (2, 10, 512), which is viewed as (2, 10, 8, 64) and transposed. Attention scores QK^T: (2, 8, 10, 10), one 10 x 10 matrix per head per sequence, the only tensor quadratic in sequence length. After softmax, multiplying by V and merging the heads, the shape returns to (2, 10, 512) and stays there through every block. Final logits from the LM head: (2, 10, 32000), one score per vocabulary entry at every position; at inference only the last position is sampled."
sources: [{"title": "Andrej Karpathy: nanoGPT model.py", "url": "https://github.com/karpathy/nanoGPT/blob/master/model.py", "note": "Read the shapes in real code."}, {"title": "Jay Alammar: The Illustrated GPT-2", "url": "https://jalammar.github.io/illustrated-gpt2/", "note": "Matrix shapes at each stage, with pictures."}, {"title": "Hugging Face: Llama model code", "url": "https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py", "note": "A production decoder with GQA and RoPE."}]
tags: ["Exercise"]
rubric: ["Embedding shape (2, 10, 512)", "Head split (2, 8, 10, 64) with d_head = 64", "Score matrix (2, 8, 10, 10)", "Logits (2, 10, 32000)", "Notes the residual shape is restored between blocks"]
practice: ["named-dimension-batched-attention-scores", "transformer-decoder-block", "decode-attention-with-kv-cache", "causal-grouped-query-attention", "transformer-activation-memory"]
---

## Why trace shapes

Most bugs in model code are shape bugs, and most performance reasoning starts from shapes: which tensor is $T \times T$, which grows with batch, which grows with vocabulary. This exercise follows one batch through a decoder.

We use batch $B = 2$, sequence $T = 10$, $d_\text{model} = 512$, $h = 8$ heads (so $d_\text{head} = 64$), MLP size $d_\text{ff} = 2{,}048$, vocabulary $V = 32{,}000$.

## The trace

| Stage | Shape | Example |
| --- | --- | --- |
| Token ids | (B, T) | (2, 10) |
| Token embedding | (B, T, d_model) | (2, 10, 512) |
| Q, K, V projections | (B, T, d_model) each | (2, 10, 512) |
| Split into heads | (B, h, T, d_head) | (2, 8, 10, 64) |
| Scores $QK^\top$ | (B, h, T, T) | (2, 8, 10, 10) |
| Softmax(scores) × V | (B, h, T, d_head) | (2, 8, 10, 64) |
| Merge heads + $W_O$ | (B, T, d_model) | (2, 10, 512) |
| MLP up | (B, T, d_ff) | (2, 10, 2048) |
| MLP down | (B, T, d_model) | (2, 10, 512) |
| After all blocks + final norm | (B, T, d_model) | (2, 10, 512) |
| LM head | (B, T, V) | (2, 10, 32000) |

~~~dg-flow The shape stays (B, T, d_model) between blocks; only attention scores and logits are different.
(B, T) -(embed)-> (B, T, 512) -(heads)-> (B, 8, T, 64) -(QK^T)-> *(B, 8, T, T) -(x V, merge)-> (B, T, 512) -(LM head)-> (B, T, 32000)
~~~

Set your own batch, length and width below and follow the shape through every stage.

~~~dg-raw
%%widget shapes Tensor shapes through embedding, attention, MLP and the output head.
~~~

## What the shapes tell you

- **The residual stream never changes shape.** (B, T, d_model) goes into every block and comes out of every block. That is what makes blocks stackable and residual adds possible ([Transformer architecture](lesson:aieng:04.03)).
- **The score tensor (B, h, T, T) is the only one quadratic in T.** It is the memory problem FlashAttention avoids by never storing it ([FlashAttention](lesson:ieng:14.03)).
- **Logits (B, T, V) are huge at the end.** With a 128k vocabulary that is 128,000 numbers per position; libraries compute the loss in chunks rather than materialise them, and inference only needs the last position.
- **Parameter count comes from weight shapes, not activations.** Here $W_Q, W_K, W_V, W_O$ are each 512 × 512 and the MLP weights are 512 × 2,048 and 2,048 × 512, whatever B and T are.

<div class="ex">
<div class="lab">Worked example</div>

**where training memory goes at real sizes**

Scale up to a Llama-3-8B-like model: $B = 8$, $T = 8{,}192$, $d_\text{model} = 4{,}096$, 32 heads, $V = 128{,}256$.

- **Residual stream** for one layer, bf16: $8 \times 8{,}192 \times 4{,}096 \times 2$ bytes = 0.54 GB.
- **Score tensor** for one layer, if stored in bf16: $8 \times 32 \times 8{,}192^2 \times 2$ bytes = 34 GB. Sixty-four times the residual stream, for a single layer: this is why fused attention kernels exist.
- **Logits** in fp32 for the loss: $8 \times 8{,}192 \times 128{,}256 = 8.4$ billion values, 34 GB. As large as the score tensor, from one matrix at the very end.

Two tensors you never see in the architecture diagram are each larger than a GPU's spare memory. Both are handled by never materialising them whole: tiled attention and chunked cross-entropy.

</div>

## With GQA and a KV cache

In a Llama-style model with fewer KV heads $g$ than query heads $h$, K and V are (B, g, T, d_head) and each is shared by $h / g$ query heads ([Multi-head attention](lesson:aieng:05.02)). In decode with a KV cache the new token has T = 1:

```text
x          [B, 1, d_model]
Q          [B, h, 1, d_head]
K, V cache [B, g, T_past + 1, d_head]   (g KV heads, each shared by h / g query heads)
scores     [B, h, 1, T_past + 1]
output     [B, 1, d_model] -> logits [B, 1, vocab]
```

Scores are now linear in context, not quadratic, and the work per step is a thin matrix-vector product rather than a matrix-matrix product. That is why decode is bound by memory bandwidth (reading weights and cache) rather than compute, and why batching many users together is so effective ([The KV cache](lesson:aieng:04.04)).

## Common mistakes

- **Not repeating KV heads.** Implement GQA and forget to expand 8 KV heads to match 32 query heads, and the matmul of (B, 32, T, 128) by (B, 8, 128, T) fails with a shape error, or worse, broadcasts silently in a library that allows it.
- **Reshape without transpose.** Going from (B, T, d_model) to (B, h, T, d_head) needs a view to (B, T, h, d_head) and then a transpose; a direct view scrambles positions across heads and still runs.
- **Computing logits for every position at inference.** Only the last row is sampled; projecting all T rows through a 128k vocabulary wastes memory and time.

Writing the expected shape next to every line, as in the trace above, catches all three before any training run.

<div class="take">
<div class="lab">Key takeaway</div>

(B, T) to (B, T, d) to (B, h, T, d_head) to (B, h, T, T) and back to (B, T, d), then (B, T, vocab). Knowing where $T \times T$ and the vocabulary appear tells you where memory goes.

</div>
