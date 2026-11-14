---
id: "05.01"
course: "ai-engineering-course"
module: "05"
slug: "masked-attention"
title: "Masked attention"
keyIdeas: ["In a decoder, token i must not see tokens after i, or training would let it cheat by reading the answer.", "Causal masking sets scores for future positions to minus infinity before softmax, so their weights become 0.", "The mask is lower-triangular: row i allows columns 0..i.", "With the mask, one forward pass trains next-token prediction at every position in parallel.", "Padding masks are separate: they hide padding tokens in batched sequences of different lengths.", "At inference with a KV cache, the new token only attends to past tokens anyway, so no explicit mask is needed for decode."]
question: "Write the 4x4 causal mask for a 4-token sequence and explain how it is applied and why it makes parallel training possible."
answer: "Mask (0 = allowed, -inf = blocked): row0 [0,-inf,-inf,-inf], row1 [0,0,-inf,-inf], row2 [0,0,0,-inf], row3 [0,0,0,0]. It is added to the scaled QK^T scores before softmax, so future positions get weight 0. Since each output position only depends on earlier tokens, every position is a valid next-token prediction, so one forward pass trains all n positions in parallel without leaking the answer."
sources: [{"title": "Jay Alammar: The Illustrated GPT-2", "url": "https://jalammar.github.io/illustrated-gpt2/", "note": "Masked self-attention with diagrams."}, {"title": "Andrej Karpathy: Let's build GPT from scratch (video)", "url": "https://www.youtube.com/watch?v=kCc8FmEb1nY", "note": "Implements the causal mask with a lower-triangular matrix."}, {"title": "PyTorch: scaled_dot_product_attention", "url": "https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html", "note": "is_causal flag and attention masks in practice."}]
---

## The cheating problem

In training, the model sees a whole sequence and must predict token i+1 from tokens 0..i. Plain self-attention lets every token look at every other, including future ones. Token 2 could simply attend to token 3 and copy it. Loss would drop, and the model would learn nothing usable for generation, where the future doesn't exist yet.

## The causal mask

Before softmax, add a mask to the n x n score matrix: 0 where attention is allowed, minus infinity where it isn't. Softmax of minus infinity is 0, so future tokens get zero weight.

```
          k0    k1    k2    k3
q0   [    0   -inf  -inf  -inf ]
q1   [    0     0   -inf  -inf ]
q2   [    0     0     0   -inf ]
q3   [    0     0     0     0  ]
```

~~~dg-flow Where the mask goes inside attention.
Q K^T / sqrt(d) -(+ causal mask)-> *Future scores = -inf -(softmax)-> Weights with zeros above diagonal -(x V)-> Output
~~~

Karpathy's GPT video builds this with a lower-triangular matrix of ones (torch.tril) and masked_fill. In PyTorch today you just pass is_causal=True to scaled_dot_product_attention.

## Why this gives parallel training

With the mask, row i of the output depends only on tokens 0..i. So the output at every position is a valid prediction of its next token, and one forward pass yields n training targets. This is the main reason transformers train so much faster than RNNs on the same hardware.

## Other masks

- **Padding mask**: batches hold sequences of different lengths padded to the same size; the mask hides pad tokens.
- **Document masks**: when several short documents are packed into one training sequence, a block-diagonal mask stops attention across documents.
- **Encoder models (BERT)**: no causal mask; every token sees both directions, which is why they're good for embeddings but not for generation.
- **Sliding-window mask**: allow only the last W positions (Mistral, Gemma local layers).

## At inference

During decode with a KV cache, there is only one new query, and the cache only holds the past. There is nothing in the future to hide, so decode needs no explicit causal mask. Prefill of a prompt still uses it.

## Compare the masks

A mask is a matrix of allowed (query, key) pairs. Disallowed pairs get −∞ added to their score before softmax, which makes their weight exactly zero. Switch between the common masks below.

~~~dg-raw
%%widget mask Causal, bidirectional, sliding-window, packed-document and padding masks.
~~~

What each is for:

- **Causal**: decoders (GPT, Llama, Claude). Token i sees tokens 0 to i. This is what lets one forward pass train on every position at once without leaking the answer.
- **Bidirectional**: encoders (BERT, embedding models). Every token sees every token, so the model can use both sides of a word. It can't generate left to right, which is why embedding models and chat models are different models.
- **Sliding window**: each token sees only the last w tokens. Bounded cost per token (07.02).
- **Packed documents**: training batches concatenate short documents into one long sequence to avoid padding waste. A block-diagonal causal mask stops document 2 from attending to document 1.
- **Padding**: in a batch of different-length sequences, the shorter ones are padded. Pad positions are masked so no real token attends to them.

## Why causal masking gives "free" training examples

A 4,096-token training sequence with a causal mask is 4,096 next-token prediction problems solved in one pass: predict token 2 from token 1, token 3 from tokens 1 and 2, and so on, all in parallel. The loss is the average over every position. Without the mask, position 3 could simply read token 4 and the model would learn to copy instead of predict.

<div class="ex">
<div class="lab">Worked example</div>

**the off-by-one that ruins a training run**

You build inputs and labels from the same token array and forget to shift labels by one. Each position is now asked to predict its own token, which it can see.

Training loss drops to near zero in a few hundred steps, and the model generates nonsense. A loss that looks too good is a masking or shifting bug until proven otherwise.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

the causal mask sets future scores to minus infinity before softmax, so each position sees only its past. That keeps training honest and lets one pass train every position at once.

</div>
