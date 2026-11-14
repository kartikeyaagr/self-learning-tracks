---
id: "04.05"
course: "ai-engineering-course"
module: "04"
slug: "what-attention-changes"
title: "What attention changes"
keyIdeas: ["Before transformers, RNNs read text one token at a time and squeezed everything into a fixed-size hidden state.", "That made long-range dependencies hard (information fades) and training slow (no parallelism across positions).", "Bahdanau et al. (2014) added attention to RNN translation so the decoder could look back at any input word.", "Transformers (2017) dropped recurrence entirely: every token attends directly to every other, in one parallel step.", "This enabled training on GPUs at huge scale, and behaviours like in-context learning via 'induction heads'.", "The price: compute and memory grow quadratically with context length, and a KV cache that grows with every token."]
question: "Name two limits of RNNs that attention removed, and one new cost that attention introduced."
answer: "Attention removed the RNN's fixed-size hidden-state bottleneck (any token is one step from any other, so long-range dependencies don't fade) and the sequential training constraint (all positions are processed in parallel, enabling large-scale GPU training). The new cost: compute and memory quadratic in sequence length, plus a KV cache that grows with every token at inference."
sources: [{"title": "Bahdanau et al.: Neural Machine Translation by Jointly Learning to Align and Translate (2014)", "url": "https://arxiv.org/abs/1409.0473", "note": "Attention added to RNNs, before transformers."}, {"title": "Lilian Weng: Attention? Attention!", "url": "https://lilianweng.github.io/posts/2018-06-24-attention/", "note": "History and variants of attention."}, {"title": "Anthropic: In-context learning and induction heads", "url": "https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html", "note": "Attention heads that copy patterns, behind in-context learning."}]
---

## Before attention

Recurrent neural networks (RNNs, LSTMs) read a sequence one step at a time, updating a hidden state vector. For translation, an encoder RNN read the whole source sentence into one final vector, and a decoder RNN generated the translation from it.

Two problems followed:

- **Bottleneck**: a 50-word sentence had to fit in one fixed-size vector. Long sentences translated badly.
- **Sequential training**: step t needs step t-1, so you can't parallelise over positions. GPUs sat mostly idle.

## Attention as a fix (2014)

Bahdanau et al. let the decoder look back at **all** encoder states at each output step and take a weighted mix of them. Translation of long sentences improved sharply, and the attention weights showed sensible word alignments. Lilian Weng's post traces this history and the variants that followed.

## Attention as the whole model (2017)

"Attention Is All You Need" removed recurrence. Every token attends to every other token directly in each layer.

| Aspect | RNN / LSTM | Transformer |
| --- | --- | --- |
| Path between distant tokens | O(n) steps | 1 step |
| Training parallelism | Sequential over positions | Fully parallel |
| Memory of the past | Fixed-size hidden state | All previous tokens (KV cache) |
| Cost per layer | O(n) | O(n^2) |
| Inference state | Constant size | Grows with context |

## What it enabled

- **Scale**: parallel training made it practical to train on trillions of tokens, which is what scaling laws needed.
- **Long-range use of context**: any earlier token is one hop away.
- **In-context learning**: Anthropic's interpretability work found "induction heads", attention circuits that spot a pattern earlier in the context ([A][B] ... [A]) and predict what followed it ([B]). These appear at the same point in training as in-context learning ability.

~~~dg-flow An induction head completes a pattern seen earlier in the context.
"Mr Dursley ... Mr" -(attend to token after earlier "Mr")-> *"Dursley" -(copy)-> Predict "Dursley"
~~~

## What it costs

Attention's direct access is also its weakness. Every token compares with every other, so cost grows with n^2, and the KV cache grows with every token generated. Most of modules 4 to 7 is about paying this bill: KV caches, GQA, paged attention, FlashAttention, sparse attention. Newer recurrent-style models (state space models like Mamba) try to bring back constant-size state, usually mixed with some attention layers.

## Sequential versus parallel, in numbers

An RNN processes a sequence one step at a time: step 500 can't start until step 499 is done. Attention computes every pair of positions at once. The figure plots both as sequence length grows.

~~~dg-raw
%%widget complexity Sequential steps for an RNN versus token pairs for attention, by sequence length.
~~~

This is the whole trade in one picture:

- **RNN**: n sequential steps, little total work, but the steps can't be spread across thousands of GPU cores. Information from token 1 has to survive n − 1 updates to reach token n.
- **Attention**: about n²/2 pairs for causal attention, a lot of total work, but all of it runs in parallel, and token n reads token 1 directly in one step.

GPUs are built for exactly the second kind of work. That is why the 2017 Transformer trained in 3.5 days on 8 GPUs to beat recurrent models that took far longer, and why training scaled to trillions of tokens afterwards.

## The quadratic bill

n² is fine at 2,000 tokens (4 million pairs per head per layer) and painful at 128,000 (16 billion). Three lines of work came out of that bill, all covered later in the course:

1. **Do the same math with less memory traffic**: FlashAttention (06.03). Still n² operations, far faster in practice.
2. **Skip most pairs**: sparse and sliding-window attention (07.02).
3. **Replace attention in some layers**: state-space models such as Mamba and hybrid models mix linear-time layers with a few attention layers. They are an active research area; most frontier models still rely mainly on attention.

<div class="ex">
<div class="lab">Worked example</div>

**why long prompts are slow to start**

Doubling a prompt from 32k to 64k tokens doubles the MLP work in prefill but quadruples the attention-score work.

At short lengths the MLP dominates, so prefill time grows roughly linearly. Past tens of thousands of tokens, attention starts to dominate and time to first token grows faster than the prompt.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

attention replaced a fixed-size memory with direct access to every token, unlocking parallel training and in-context learning. The cost is quadratic compute and a cache that grows with context.

</div>
