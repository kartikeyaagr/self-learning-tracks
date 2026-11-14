---
id: "16.07"
course: "inference-eng-course"
module: "16"
slug: "dflash-dynamic-speculation"
title: "DFlash & Dynamic Speculation"
keyIdeas: ["Drafting costs time too; a drafter that walks positions serially adds latency every round.", "Block drafters fill several positions in one pass to keep drafting cost flat.", "Conditioning the drafter on the target's hidden features improves block drafts.", "Draft length should be a runtime decision, based on confidence and load.", "Reported speedups depend heavily on the setting: batch size, temperature, task."]
question: "What is the trade-off between a drafter that proposes positions one at a time and one that fills a whole block in a single pass, and how does dynamic draft length help?"
answer: "A sequential drafter conditions each guess on the previous one, so later guesses are more accurate, but it adds k serial steps to every round. A block (parallel) drafter fills all k positions in one pass, so drafting latency is flat, but guesses are less coordinated and acceptance can drop for later positions. Dynamic speculation picks the draft length per step or per request from confidence and load, drafting long when guesses are likely accepted and the GPU has spare compute, short or not at all otherwise."
sources: [{"title": "Optimizing Speculative Decoding for Serving Large Language Models Using Goodput", "url": "https://arxiv.org/abs/2406.14066", "note": "Choosing speculation length under load."}, {"title": "Medusa", "url": "https://arxiv.org/abs/2401.10774", "note": "Parallel heads as a block drafter."}, {"title": "fanout.sh lesson 16.07 (summary only)", "url": "https://fanout.sh/inference-eng/curriculum/16-07", "note": "Outline of the original lesson; the full text is paid."}]
tags: ["speculative"]
---

## A note on sources

The original course lesson on DFlash is behind a paywall, and the details of the DFlash method itself could not be verified for this rewrite. This lesson covers the ideas its outline lists: the drafter's own cost, drafting a block at once, conditioning the drafter on the target, and treating draft length as a decision. Check the original paper before relying on any DFlash-specific claim.

## The drafter's own loop

A speculative round costs drafting time plus verification time. With a sequential drafter, drafting k tokens takes k small passes, one after another. Even if each is cheap, k serial steps add up, and they matter more as the target gets faster.

<div class="ex">
<div class="lab">Worked example</div>

**when drafting dominates**

A target pass takes 8 ms. A draft model takes 1 ms per token. With k = 8, drafting takes 8 ms, as long as verification.

Expected tokens per round at acceptance 0.8 and k = 8: about 4.3. Time per round: 16 ms. Time per token: 3.7 ms, a 2.2 times speedup.

If drafting the 8 tokens took 1 ms in total instead of 8 ms, the round would take 9 ms and time per token would be 2.1 ms, a 3.8 times speedup with the same acceptance.

</div>

## Filling a block instead of walking it

A **block drafter** predicts positions t + 1 to t + k in one pass, as Medusa's heads do, instead of one at a time. The drafting cost then stays roughly constant as k grows, which makes longer drafts affordable.

The risk is accuracy. Each position is predicted without seeing the drafter's own guesses for the positions before it, so later positions are guessed less accurately, and acceptance falls with position faster than with a sequential drafter. Methods in this family try to close that gap, for example with iterative refinement of the block or with diffusion-style denoising of all positions together.

## Conditioning on the target

A drafter that sees the target's hidden states starts with much more information than a drafter that sees only tokens. EAGLE uses the target's features. A block drafter can do the same, reading features from one or several layers of the target. This narrows the accuracy gap of block drafting, because the target's features already encode a lot about the next several tokens.

## Read speedup claims with their setting

Published speedups for speculative methods depend heavily on the conditions:

- **Batch size**: almost always 1, the best case for speculation.
- **Temperature**: often greedy, which gives the highest acceptance.
- **Task**: code and math benchmarks flatter most methods.
- **Hardware**: GPUs with more spare compute show bigger gains.

Compare methods in your own setting, at your own batch sizes and sampling settings, before choosing.

## Draft length as a decision

Rather than a fixed k, a drafter can decide how far to draft each round:

| Signal | Effect on k |
| --- | --- |
| High drafter confidence | draft longer |
| Recent low acceptance for this request | draft shorter |
| Large running batch | shorter, or no speculation |
| Latency-critical request | longer |

Dynamic schemes stop drafting when the drafter's confidence falls below a threshold, or choose k to maximize expected tokens per unit of time given the current costs. They usually beat any fixed k across a mixed workload.

## Deep dive: where the draft cost c comes from

The drafter runs ⟦m865⟧ sequential steps of its own. Each one is memory-bound and pays its own fixed overhead:

⟦M864⟧

~~~dg-raw
%%bars Draft cost per token relative to an 8B target step of 6.75 ms (estimates).
unit = × target step
1B draft model, eager (3 ms overhead) | 0.55 | c1
1B draft model, CUDA graphs | 0.12 | c2
One-layer head reusing the target's LM head | 0.08 | c3
Prompt lookup (n-gram on the CPU) | 0.01 | c4
%%
~~~

A separate 1B draft is 0.71 ms of weight reads, but without graphs its launch overhead is four times that. The one-layer head is small but must read the 1 GB LM head on every draft step, which sets its cost.

<div class="take">
<div class="lab">Key takeaway</div>

speculation has two costs, drafting and verification. Block drafting caps the first by predicting several positions at once, conditioning on the target's features keeps those guesses accurate, and dynamic draft length manages the second.

</div>
