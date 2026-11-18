---
id: "05.01"
course: "ai-engineering-course"
module: "05"
slug: "masked-attention"
title: "Masked attention"
keyIdeas: ["In a decoder, token i must not see tokens after i, or training would let it cheat by reading the answer.", "Causal masking adds minus infinity to future scores before softmax, so their weights become exactly 0 and each row still sums to 1.", "With the mask, one forward pass trains next-token prediction at every position in parallel.", "Padding, packed-document and sliding-window masks are separate and combine by adding.", "Single-token decode with a KV cache needs no causal mask, but prefill, batched decode and multi-token steps (chunked prefill, speculative verification) do."]
question: "Write the 4x4 causal mask for a 4-token sequence and explain how it is applied and why it makes parallel training possible."
answer: "Mask (0 = allowed, -inf = blocked): row 0 [0, -inf, -inf, -inf], row 1 [0, 0, -inf, -inf], row 2 [0, 0, 0, -inf], row 3 [0, 0, 0, 0]. It is added to the scaled QK^T scores before softmax, so future positions get weight exactly 0 and each row of weights still sums to 1; zeroing weights after softmax would leave rows that no longer sum to 1. Because each output position then depends only on itself and earlier tokens, the output at every position is an honest prediction of the next token, so one forward pass over an n-token sequence trains all n predictions in parallel without leaking the answer, which an RNN cannot do."
sources: [{"title": "Jay Alammar: The Illustrated GPT-2", "url": "https://jalammar.github.io/illustrated-gpt2/", "note": "Masked self-attention with diagrams."}, {"title": "Andrej Karpathy: Let's build GPT from scratch (video)", "url": "https://www.youtube.com/watch?v=kCc8FmEb1nY", "note": "Implements the causal mask with a lower-triangular matrix."}, {"title": "PyTorch: scaled_dot_product_attention", "url": "https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html", "note": "is_causal flag and attention masks in practice."}]
rubric: ["Writes a lower-triangular mask with 0 on and below the diagonal and -inf above", "Adds the mask to the scores before softmax, not to the weights after", "Explains that future weights become 0 and rows still sum to 1", "Explains that every position becomes a valid next-token target in one pass"]
practice: ["masked-causal-attention", "masked-safe-softmax", "sliding-window-causal-mask", "pad-and-mask-to-bucket-length"]
---

## The cheating problem

In training, the model sees a whole sequence and must predict token $i+1$ from tokens $0..i$. Plain self-attention lets every token look at every other, including future ones. Token 2 could simply attend to token 3 and copy it. Loss would drop, and the model would learn nothing usable for generation, where the future does not exist yet.

## The causal mask

Before softmax, add a mask $M$ to the $n \times n$ score matrix: 0 where attention is allowed, $-\infty$ where it is not.

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V, \qquad M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}
$$

Since $e^{-\infty} = 0$, future tokens get exactly zero weight and the remaining weights still sum to 1. For four tokens the mask is lower-triangular:

```text
          k0    k1    k2    k3
q0   [    0   -inf  -inf  -inf ]
q1   [    0     0   -inf  -inf ]
q2   [    0     0     0   -inf ]
q3   [    0     0     0     0  ]
```

~~~dg-flow Where the mask goes inside attention.
Q K^T / sqrt(d) -(+ causal mask)-> *Future scores = -inf -(softmax)-> Weights with zeros above diagonal -(x V)-> Output
~~~

Karpathy's GPT video builds this with a lower-triangular matrix of ones (`torch.tril`) and `masked_fill`. In PyTorch today you pass `is_causal=True` to `scaled_dot_product_attention`, and fused kernels such as FlashAttention skip the masked tiles entirely instead of computing and discarding them.

<div class="ex">
<div class="lab">Worked example</div>

**one masked row, done right and done wrong**

Query $q_2$ has scaled scores $[1.0, 2.0, 0.5, 3.0]$ against keys 0 to 3. Key 3 is in its future.

- **Mask before softmax** (correct): the row becomes $[1.0, 2.0, 0.5, -\infty]$. Softmax over the first three gives weights $[0.231, 0.629, 0.140]$, which sum to 1, and key 3 gets 0.
- **Zero after softmax** (wrong): softmax over all four gives $[0.085, 0.232, 0.052, 0.631]$. Zero the last and the row keeps $[0.085, 0.232, 0.052]$, which sums to only 0.369. The output is shrunk to about a third of its proper size, and the ratios were set by a score the token should never have seen.

Key 3 carried the largest score, so before masking it would have taken 63% of the weight: exactly the cheat the mask exists to block.

</div>

## Why this gives parallel training

With the mask, row $i$ of the output depends only on tokens $0..i$, so the output at every position is a valid prediction of its next token. A 4,096-token training sequence is 4,096 next-token problems solved in one pass: predict token 1 from token 0, token 2 from tokens 0 and 1, and so on, with the loss averaged over every position. This is the main reason transformers train so much faster than RNNs on the same hardware ([What attention changes](lesson:aieng:04.05)).

## Other masks

A mask is just a matrix of allowed (query, key) pairs. Switch between the common ones below.

~~~dg-raw
%%widget mask Causal, bidirectional, sliding-window, packed-document and padding masks.
~~~

| Mask | Allowed pairs | Used for |
| --- | --- | --- |
| Causal | Key position $\le$ query position | Decoders: GPT, Llama, Claude |
| Bidirectional | All pairs | Encoders: BERT and embedding models, which use both sides of a word but cannot generate left to right; that is why embedding and chat models are different models |
| Sliding window | The last $W$ positions only | Local layers in Mistral, Gemma 3, gpt-oss; caps the cache ([The KV cache](lesson:aieng:04.04)) |
| Packed documents | Block-diagonal causal | Training: short documents concatenated into one sequence to avoid padding waste, without document 2 attending to document 1 |
| Padding | Real tokens only | Batches of different-length sequences, padded to one size |

Masks combine by adding: a packed batch with padding uses causal, document and padding masks together. Masks on the **loss** are a separate thing: chat fine-tuning computes loss only on assistant tokens, while attention still sees the whole conversation.

## At inference

During decode with a KV cache, a single new query attends to cached past tokens, so there is nothing in the future to hide and the causal part needs no explicit mask. Three cases still need one:

- **Prefill** of the prompt runs all prompt positions at once and uses the causal mask.
- **Batched decode** over sequences of different lengths needs a length or padding mask so a query never reads another sequence's empty slots.
- **Several new tokens per step**, as in chunked prefill or verifying a speculative draft, need a causal mask within the new block, offset by the number of tokens already cached.

## Common mistakes

- **Forgetting to shift labels.** Build inputs and labels from the same array without shifting by one, and each position predicts its own token, which it can see. Loss drops to near zero in a few hundred steps and the model generates nonsense. A loss that looks too good is a masking or shifting bug until proven otherwise.
- **Masking after softmax**, as in the worked example.
- **Fully masked rows.** A row with every score at $-\infty$ (a pure padding query) gives zero divided by zero, which is NaN, and NaNs spread through the batch. Use a safe softmax that returns zeros for empty rows, or never compute those rows.
- **Using a finite "large negative" in fp16.** $-10^9$ overflows fp16; use $-\infty$ or the dtype's minimum.
- **Off-by-one in the cache offset.** With $c$ cached tokens, new query $i$ may see keys up to $c + i$, not $i$.

<div class="take">
<div class="lab">Key takeaway</div>

The causal mask adds $-\infty$ to future scores before softmax, so each position sees only its past. That keeps training honest, lets one pass train every position at once, and comes back at inference whenever more than one new token is processed per step.

</div>
