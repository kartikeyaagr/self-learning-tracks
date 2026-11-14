---
id: "00.03"
course: "inference-eng-course"
module: "00"
slug: "dot-products-softmax-attention-math"
title: "Dot Products, Softmax & Attention Math"
keyIdeas: ["A dot product measures how aligned two vectors are. Attention scores are dot products between a query and every key.", "Softmax turns scores into weights that are positive and sum to 1.", "Scaled dot-product attention: softmax(Q K^T / sqrt(d)) V.", "Subtract the row max before exp for numerical safety. This trick also enables online softmax, used by FlashAttention.", "A causal mask sets future positions to minus infinity so a token only attends to earlier ones."]
question: "Why does attention divide scores by the square root of the head dimension, and why do implementations subtract the max before softmax?"
answer: "Dot products of random vectors grow with dimension, so dividing by sqrt(d) keeps score variance near 1 and stops softmax saturating into a near one-hot distribution. Subtracting the max keeps exp() from overflowing; it does not change the result because softmax is invariant to adding a constant to every input."
sources: [{"title": "Attention Is All You Need", "url": "https://arxiv.org/abs/1706.03762", "note": "The original scaled dot-product attention."}, {"title": "3Blue1Brown: Attention in transformers", "url": "https://www.3blue1brown.com/lessons/attention", "note": "Visual explanation of queries, keys and values."}]
tags: ["math", "attention"]
---

## The dot product as similarity

The dot product of two vectors multiplies them element by element and adds the results. For vectors a and b of length d, `a · b = a₁b₁ + a₂b₂ + … + a_d b_d`. If two vectors point the same way, the dot product is large and positive. If they are unrelated, it is near zero. Attention is built on this: a query vector "asks" for something, key vectors "advertise" what each token has, and the dot product scores the match.

## Softmax turns scores into weights

Softmax maps a list of scores to probabilities that sum to 1:

softmax(x)ᵢ = exp(xᵢ) ÷ Σⱼ exp(xⱼ)

Big scores get most of the weight. A score 2 higher than another gets e² ≈ 7.4× more weight.

<div class="ex">
<div class="lab">Worked example</div>

**softmax by hand**

Scores [2.0, 1.0, 0.1]. exp gives [7.39, 2.72, 1.11]. Sum = 11.22.

Softmax = [0.659, 0.242, 0.099]. The first item gets two thirds of the weight.

In practice you subtract the max first (scores become [0, −1, −1.9]) so exp never overflows. The result is identical.

</div>

## Attention in four lines

For one head with query matrix Q `[T, d]`, keys K `[S, d]` and values V `[S, d]`:

1. Scores = Q × Kᵀ, shape `[T, S]`. Each query is dotted with every key.
2. Scale by 1 ÷ √d so the scores do not grow with dimension and saturate the softmax.
3. Apply a **causal mask**: a token may not attend to tokens after it. Masked scores are set to −∞, which softmax turns into 0.
4. Weights = softmax over S. Output = weights × V, shape `[T, d]`. Each output is a weighted average of value vectors.

~~~dg-flow Attention for one head.
Q, K -(dot products)-> Scores [T×S] -(scale, mask)-> *Softmax -(weights × V)-> Output [T×d]
~~~

## Why √d

If each element of q and k is random with variance 1, their dot product over d dimensions has variance d. With d = 128 the standard deviation is about 11, so softmax would put almost all weight on one token. Dividing by √128 ≈ 11.3 brings the variance back to 1 and keeps gradients and outputs well-behaved.

## The cost of attention

Scores are `[T, S]`. In prefill, T = S = prompt length, so the score matrix has n² entries per head. For a 32,000-token prompt with 32 heads, that is 32 × 32,000² ≈ 33 billion scores per layer. Stored in FP16, that would be 66 GB per layer, which is why real kernels (FlashAttention) never materialize it; they compute softmax in tiles. In decode, T = 1, so scores are `[1, S]`: cheap in FLOPs, but the whole K and V for S tokens must be read from memory every step. That is the KV cache read you will meet many times.

## Online softmax, a preview

Softmax needs the max and the sum over all S scores before any output is final. FlashAttention avoids storing all scores by keeping a running max and running sum, rescaling partial results when a new larger max appears. The math is exact, not an approximation. You will implement this in the attention module.

## Numerical precision

In BF16, exp overflows above about 88 and the 8-bit mantissa gives roughly 2 to 3 significant decimal digits. Kernels compute softmax in FP32 internally even when inputs and outputs are BF16. If you write your own kernels, keep the max subtraction and FP32 accumulation, or you will get NaNs on long contexts.

## Deep dive: why scores are divided by √d

Suppose the components of ⟦m76⟧ and ⟦m77⟧ are independent with mean 0 and variance 1. Then each product ⟦m78⟧ has mean 0 and variance 1, and the dot product sums ⟦m79⟧ of them:

⟦M70⟧

Without the scaling, scores for ⟦m80⟧ would have a standard deviation of about 11. Softmax of numbers that spread out is nearly one-hot, and its gradient during training is close to zero. Scaling keeps the scores in a range where softmax is smooth.

## Deep dive: softmax without overflow

Softmax is unchanged if you subtract any constant ⟦m81⟧ from all inputs:

⟦M71⟧

Choosing ⟦m82⟧ makes every exponent at most 0, so nothing overflows. In FP16, ⟦m83⟧ already exceeds the largest representable value (65,504), so this is not optional.

## Deep dive: online softmax, the recurrence

FlashAttention processes keys in blocks and never holds a full row of scores. It keeps a running maximum ⟦m84⟧, a running denominator ⟦m85⟧ and a running output ⟦m86⟧. When a new block of scores ⟦m87⟧ with values ⟦m88⟧ arrives:

⟦M72⟧

⟦M73⟧

⟦M74⟧

At the end the answer is ⟦m89⟧. The factor ⟦m90⟧ rescales what was accumulated under the old maximum, so the result is exactly the same as a full softmax.

<div class="ex">
<div class="lab">Worked example</div>

**two blocks by hand**

Scores [1, 3] then [2, 5], with values [10, 20] then [30, 40].

Block 1: ⟦m91⟧, ⟦m92⟧, ⟦m93⟧.

Block 2: ⟦m94⟧, rescale factor ⟦m95⟧. ⟦m96⟧. ⟦m97⟧.

Result ⟦m98⟧. A full softmax over [1, 3, 2, 5] gives weights [0.015, 0.112, 0.041, 0.831] and the same weighted sum, 36.9.

</div>

~~~dg-raw
%%grid Attention weights for a 5-token causal example (each row sums to 1).
heat = 1
cols = The, cat, sat, on, mat
The | 1.00 . . . .
cat | 0.38 0.62 . . .
sat | 0.12 0.61 0.27 . .
on | 0.08 0.21 0.48 0.23 .
mat | 0.05 0.31 0.14 0.29 0.21
%%
~~~

## Deep dive: what attention costs, in symbols

For one layer with ⟦m99⟧ query heads of size ⟦m100⟧ and a context of ⟦m101⟧ tokens, a single new query does ⟦m102⟧ FLOPs for ⟦m103⟧ and the same for ⟦m104⟧, per head:

⟦M75⟧

It must also read the cached K and V: ⟦m105⟧ values. Intensity is therefore about ⟦m106⟧ FLOPs per byte, which is 4 for Llama 3 8B. Decode attention is memory bound at any batch size, because each sequence has its own KV.

<div class="take">
<div class="lab">Key takeaway</div>

attention is dot-product scoring, a scaled and masked softmax, and a weighted average of values. Prefill attention costs grow with n², decode attention cost is mostly reading the KV cache.

</div>
