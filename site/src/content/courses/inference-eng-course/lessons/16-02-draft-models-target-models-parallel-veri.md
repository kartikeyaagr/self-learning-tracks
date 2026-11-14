---
id: "16.02"
course: "inference-eng-course"
module: "16"
slug: "draft-models-target-models-parallel-verification"
title: "Draft Models, Target Models & Parallel Verification"
keyIdeas: ["A draft model (small, same tokenizer) proposes k tokens.", "The target model scores all k plus one more in one pass.", "Accept draft tokens while they pass a probabilistic test; stop at first rejection.", "Each round yields 1 to k+1 tokens; output distribution equals the target's.", "Speedup depends on acceptance rate and the draft's cost relative to the target."]
question: "Walk through one round of draft-model speculative decoding with k = 4."
answer: "The small draft model generates 4 tokens autoregressively from the current context (4 cheap passes). The target model runs one forward pass on the context plus those 4 tokens, producing distributions for 5 positions. Starting from the first, each draft token is accepted or rejected by comparing draft and target probabilities. At the first rejection, a corrected token is sampled and the rest discarded; if all 4 are accepted, a fifth token is sampled from the target's last distribution. The round yields between 1 and 5 tokens."
sources: [{"title": "Fast Inference from Transformers via Speculative Decoding", "url": "https://arxiv.org/abs/2211.17192", "note": "The algorithm and proof."}, {"title": "vLLM: Speculative decoding", "url": "https://docs.vllm.ai/en/latest/features/spec_decode.html", "note": "How engines run it."}]
tags: ["speculative"]
---

## One round of draft and verify

The classic form of speculative decoding uses two models: a small **draft** model and the large **target** model you actually want outputs from.

~~~dg-seq One round with a draft length of 4. Press Play to step through it.
Draft > Draft : 4 small passes
Draft > Target : context + t1..t4
Target > Target : 1 pass, 5 outputs
Target -> Engine : accept t1, t2
Target -> Engine : replace t3
~~~

1. The draft model generates k = 4 tokens autoregressively: t1, t2, t3, t4. Each is one cheap forward pass.
2. The target model runs one pass over the context plus t1 to t4. Because it processes all positions at once, it produces 5 next-token distributions: one after the context, and one after each draft token.
3. Compare position by position. Suppose t1 and t2 match, and t3 does not. Accept t1 and t2, and take the target's own token at position 3 instead of t3. Discard t4.
4. The round's output is t1, t2 and the corrected token: three tokens for one target pass and four draft passes.

If all k guesses are accepted, the target's distribution at position k + 1 gives a bonus token, so a round can produce up to k + 1 tokens.

## Requirements

- **Same tokenizer and vocabulary.** The target must score exactly the tokens the draft proposed.
- **KV cache bookkeeping.** The target's verification pass writes K and V for all k positions. Rejected positions must be rolled back, which with paged KV means trimming the sequence's length and possibly freeing a block.
- **The draft keeps its own KV cache**, which also needs memory and must be rolled back on rejection.
- **A good draft model.** It must agree with the target often and cost much less. A 1B model drafting for a 70B model from the same family is a typical pairing.

## Expected tokens per round

Assume each draft token is accepted with probability a, independently. A round stops at the first rejection. The expected number of tokens per round, including the corrected or bonus token, is:

expected tokens = (1 − a^(k+1)) ÷ (1 − a)

| Acceptance a | k = 4 | k = 8 |
| --- | --- | --- |
| 0.6 | 2.2 | 2.5 |
| 0.8 | 3.4 | 4.3 |
| 0.9 | 4.1 | 6.1 |

Gains flatten as k grows, because later positions are reached only if every earlier one was accepted.

## The speedup

Each round costs one target pass plus k draft passes. If a draft pass costs c times a target pass:

speedup ≈ expected tokens ÷ (1 + k × c)

This ignores the small extra cost of verifying k + 1 positions instead of 1.

~~~dg-raw
%%widget spec Change acceptance, draft length and draft cost. Watch the speedup peak and then fall as k grows.
a = 0.8
k = 4
c = 0.1
%%
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a 70B target with a 1B draft**

A batch-1 pass of the 70B model on its GPUs takes 20 ms. The 1B draft takes 2 ms per token, so c = 0.1.

Measured acceptance on a chat workload: a = 0.8. With k = 4, expected tokens per round = 3.4.

Cost per round: 20 ms + 4 × 2 ms = 28 ms. Time per token: 28 ÷ 3.4 ≈ 8.2 ms, against 20 ms without speculation. Speedup about 2.4 times.

With k = 8: 4.3 tokens per round, cost 36 ms, 8.4 ms per token. No better, because the extra drafts are rarely all accepted.

</div>

## When a draft model is a poor fit

- No small model shares the target's tokenizer.
- The draft's memory and KV cache compete with the target for GPU memory.
- The draft's own sequential passes become a large part of the round when the target is fast.

These limits motivate the drafter-free and head-based methods later in this module.

## Deep dive: deriving the expected tokens per round

Assume each drafted token is accepted with probability ⟦m829⟧, independently, and the round stops at the first rejection. Then position ⟦m830⟧ is accepted with probability ⟦m831⟧, and the target always adds one token of its own (the correction or the bonus):

⟦M827⟧

With draft cost ⟦m832⟧ per token, relative to one target step, and free verification, the speed-up over plain decoding is

⟦M828⟧

~~~dg-raw
%%chart Speed-up against draft length at batch 1, draft cost c = 0.05.
x = 1, 2, 3, 4, 6, 8, 12
xlab = Draft length k
ylab = Speed-up
α = 0.6: 1.52, 1.78, 1.89, 1.92, 1.87, 1.77, 1.56
α = 0.8: 1.71, 2.22, 2.57, 2.80, 3.04, 3.09, 2.95
α = 0.9: 1.81, 2.46, 2.99, 3.41, 4.01, 4.38, 4.66
%%
~~~

The expected-token sum saturates at ⟦m833⟧: 2.5 tokens for ⟦m834⟧, 10 for 0.9. Beyond a few multiples of ⟦m835⟧, extra draft tokens are almost never accepted but are always paid for.

<div class="take">
<div class="lab">Key takeaway</div>

draft k tokens cheaply, verify them in one target pass, keep the accepted prefix and correct the first miss. Acceptance rate and draft cost decide the speedup, and longer drafts give diminishing returns.

</div>
