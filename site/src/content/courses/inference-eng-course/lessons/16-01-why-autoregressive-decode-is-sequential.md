---
id: "16.01"
course: "inference-eng-course"
module: "16"
slug: "why-autoregressive-decode-is-sequential"
title: "Why Autoregressive Decode Is Sequential"
keyIdeas: ["Each token depends on the previous one, so standard decode needs one forward pass per token.", "At small batch each pass is memory bound and uses little of the GPU's compute.", "Scoring several tokens in one pass costs about the same as scoring one.", "If we can guess future tokens cheaply, the big model can check many at once.", "That is speculative decoding: guess with something cheap, verify in parallel."]
question: "Why can a model verify 5 proposed tokens in about the same time it takes to generate 1?"
answer: "Decode is memory bound: a step's time is dominated by reading the weights. Feeding 5 tokens instead of 1 through the same forward pass reads the weights once and does 5 times the (still small) compute, so the step takes nearly the same time, while producing logits for all 5 positions at once."
sources: [{"title": "Fast Inference from Transformers via Speculative Decoding", "url": "https://arxiv.org/abs/2211.17192", "note": "Leviathan et al."}, {"title": "Accelerating Large Language Model Decoding with Speculative Sampling", "url": "https://arxiv.org/abs/2302.01318", "note": "Chen et al."}]
tags: ["speculative"]
---

## The sequential bottleneck

Token t + 1 depends on token t, so generating 200 tokens takes 200 forward passes of the full model, one after another. At small batch, each pass reads every weight from memory to produce a single token per sequence. For an 8B model on an H100, that sets a floor of about 5 ms per token, or about 200 tokens per second, for one user. No scheduler or kernel can go below it while producing one token per pass.

## The compute sits idle

A batch-1 decode step for an 8B model does about 16 GFLOP of work in about 5 to 6 ms. That is about 3 TFLOP/s, roughly 0.3% of the H100's dense BF16 peak. The GPU spends the step waiting for weights to arrive.

Now consider running the same pass over several tokens of the same sequence at once, like a tiny prefill. The weights are still read once. The extra tokens add only a little compute and a little KV traffic:

| Tokens processed in one pass | Relative step time (memory-bound regime) |
| --- | --- |
| 1 | 1.0 |
| 4 | about 1.05 |
| 8 | about 1.1 |
| 64 | noticeably more |
| several hundred | compute bound, roughly proportional |

So processing 4 tokens costs about the same as processing 1. The problem is that during normal decode you do not know tokens 2 to 4 yet.

~~~dg-raw
%%widget roof At batch 1 the step sits far down the memory slope. Verifying k extra tokens moves it right, using compute that was idle anyway.
mode = Decode
n = 0
%%
~~~

## The idea

If something cheap can **guess** the next few tokens, the large model can **check** all of them in one pass. The check works like prefill: feed the guessed tokens in, and the model produces its own prediction at every position at once. Wherever the guess matches what the model would have produced, that token is accepted. At the first mismatch, the model's own prediction replaces the guess, and the round ends.

~~~dg-flow Speculative decoding in one round.
Cheap proposer -(k guesses)-> *Target verifies all k in one pass -> Keep the accepted prefix -> Next round
~~~

Every correct guess saves one full forward pass of the large model. Even when every guess is wrong, the round still produces one correct token, the model's own prediction at the first position, so it is never worse than normal decoding except for the cost of the guessing and the slightly longer pass.

<div class="ex">
<div class="lab">Worked example</div>

**the best case**

The target model takes 6 ms per pass at batch 1. A proposer guesses 4 tokens, and the verification pass over 5 positions takes 6.5 ms.

If all 4 guesses are right, the round produces 5 tokens (the 4 guesses plus the model's next prediction) in 6.5 ms plus guessing time: about 1.3 ms per token instead of 6.

If the first guess is wrong, the round produces 1 token in 6.5 ms plus guessing time: slightly worse than normal.

The average depends on how often guesses are right, which the next lessons quantify.

</div>

## Why it can keep quality exactly

For greedy decoding, the check is simple: accept a guess if it equals the model's top token. The output is then exactly what greedy decoding would produce. For sampling, a careful acceptance rule (lesson 16.03) makes the output distribution identical to sampling from the large model alone. Speculative decoding changes speed, not outputs.

## Where the guesses come from

Different methods propose tokens in different ways:

- **A small draft model** from the same family, run autoregressively.
- **N-gram lookup** in the prompt and the output so far.
- **Extra heads** on the large model, such as Medusa and EAGLE.
- **Multi-token prediction heads** trained into the model itself.

The rest of this module covers each.

## Deep dive: what verification costs at each batch size

Verifying ⟦m823⟧ drafted tokens for each of ⟦m824⟧ sequences is one target step over ⟦m825⟧ tokens. With the step model from 07.04:

⟦M822⟧

While ⟦m826⟧ stays below the ridge (about 214 tokens for an 8B model on an H100), verification costs the same as a plain step. Above it, every extra drafted token costs real time.

~~~dg-raw
%%chart Predicted verification step time against draft length (Llama 3 8B, H100, KV reads ignored).
x = 0, 2, 4, 8, 16
xlab = Draft length k
ylab = ms per target step
Batch 1: 6.75, 6.75, 6.75, 6.75, 6.75
Batch 64: 6.75, 6.75, 9.0, 15.4, 28.2
Batch 256: 7.4, 20.2, 33.0, 58.6, 109.8
%%
~~~

At batch 1, sixteen extra tokens are free. At batch 256, even two cost almost three times a normal step. Everything else about speculation follows from this curve.

<div class="take">
<div class="lab">Key takeaway</div>

decode leaves most of the GPU's compute idle, because each pass reads all weights for one token. Verifying several guessed tokens in one pass uses that idle compute to skip forward passes, without changing outputs.

</div>
