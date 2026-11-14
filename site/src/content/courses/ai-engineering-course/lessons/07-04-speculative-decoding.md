---
id: "07.04"
course: "ai-engineering-course"
module: "07"
slug: "speculative-decoding"
title: "Speculative decoding"
keyIdeas: ["Decode is memory-bound: each step reads all the weights to produce one token, leaving compute idle.", "Speculative decoding: a small, fast draft model proposes k tokens; the big target model checks all k in one forward pass.", "Checking k tokens in parallel costs about the same as one decode step, because the weights are read once.", "Accept drafted tokens left to right while they pass; at the first rejection, sample a corrected token from the target and start again.", "The acceptance rule (accept with probability min(1, p_target / p_draft)) guarantees the output has exactly the target model's distribution.", "Speedup depends on acceptance rate; 2-3x is common. Variants: Medusa heads, EAGLE, n-gram / prompt lookup drafts."]
question: "Why can a big model verify 5 drafted tokens in about the same time it takes to generate 1, and how does speculative decoding keep the output distribution unchanged?"
answer: "Decode is memory-bandwidth bound: one step reads all the weights to produce one token, so a forward pass over 5 positions reads the weights once and only adds cheap compute, taking about the same time. Drafts are accepted left to right with probability min(1, p_target/p_draft) (or exact match in greedy mode); at the first rejection a replacement is sampled from the residual distribution max(0, p_target - p_draft). This rejection-sampling rule makes the output exactly distributed as the target model's."
sources: [{"title": "Leviathan et al.: Fast Inference from Transformers via Speculative Decoding (2022)", "url": "https://arxiv.org/abs/2211.17192", "note": "Draft-then-verify with the same output distribution."}, {"title": "Chen et al.: Accelerating LLM Decoding with Speculative Sampling (2023)", "url": "https://arxiv.org/abs/2302.01318", "note": "DeepMind's version, 2-2.5x on Chinchilla 70B."}, {"title": "Cai et al.: Medusa (2024)", "url": "https://arxiv.org/abs/2401.10774", "note": "Extra decoding heads instead of a separate draft model."}]
---

## Idle compute in decode

From lesson 01.03: decode generates one token per forward pass, and each pass reads every weight from GPU memory. The GPU's arithmetic units are mostly idle, waiting on memory. If we could get several useful tokens out of one pass, we would use that spare compute for free.

## Draft, then verify

Leviathan et al. (Google) and Chen et al. (DeepMind) proposed the same idea in 2022-2023:

1. A small **draft** model (or cheap method) quickly generates k candidate tokens.
2. The large **target** model runs one forward pass over the prompt plus all k drafts. Like prefill, this computes the target's distribution at every one of those positions in parallel.
3. Compare left to right. Accept drafts while they pass; at the first rejection, replace that token with one sampled from the target and discard the rest.
4. Repeat from the new position.

~~~dg-seq One speculative step with k = 4.
Draft model > Draft model: propose t1 t2 t3 t4
Draft model > Target model: verify all four in one pass
Target model > Target model: accept t1, t2; reject t3
Target model -> Draft model: t1 t2 + corrected t3', continue
~~~

Each round yields at least one token (the correction) and up to k + 1 when all drafts pass.

## Why verification is cheap

Verifying k tokens is a forward pass over k positions. Since decode is memory-bound, reading the weights once and doing k times the maths takes about as long as one normal step, for small k.

## Same output, not an approximation

For greedy decoding, accept a draft if it equals the target's top token. For sampling, the papers use a rejection rule: accept draft token x with probability min(1, p_target(x) / p_draft(x)); if rejected, sample from the normalised difference max(0, p_target - p_draft). This guarantees the final tokens follow **exactly** the target model's distribution. Quality is unchanged; only speed changes.

## What determines the speedup

- **Acceptance rate**: how often the draft agrees with the target. Higher on predictable text (code, boilerplate, structured output).
- **Draft cost**: the draft must be much faster than the target (often 10-50x smaller, same tokenizer).
- **k**: more drafts help when acceptance is high, waste work when it's low.
Chen et al. reported 2-2.5x on Chinchilla 70B.

## Variants

| Method | Draft source |
| --- | --- |
| Classic | Separate small model from the same family |
| Medusa | Extra prediction heads on the target model itself |
| EAGLE | Lightweight head predicting from the target's hidden states |
| Prompt lookup / n-gram | Copy spans from the prompt (great for editing and RAG) |

## Simulate a draft-and-verify round

A small draft model proposes k tokens. The large model checks all k in one forward pass, accepts the longest correct prefix, and adds one token of its own. Change the acceptance rate and draft length to see the speedup.

~~~dg-raw
%%widget specdec Speculative decoding rounds, expected tokens per round and speedup.
~~~

The expected number of tokens per round with acceptance rate α and draft length k (Leviathan et al., 2023):

```
E[tokens] = (1 - α^(k+1)) / (1 - α)
```

At α = 0.8 and k = 4 that is (1 − 0.8^5) / 0.2 = 3.36 tokens per target forward pass instead of 1. Subtract the cost of running the draft model k times and you get the real speedup, often 2 to 3× in practice.

## Why the output is unchanged

For greedy decoding, the target accepts a draft token only if it is the token the target would have picked. For sampling, the rule accepts draft token x with probability min(1, p_target(x) / p_draft(x)), and on rejection samples from the normalised difference max(0, p_target − p_draft). This is a form of rejection sampling, and it provably produces exactly the target model's distribution. Speculative decoding is a speed technique, not a quality trade.

## Where the speedup comes from, and where it doesn't

- Decode is memory-bound: each step reads all the weights to produce one token. Checking k tokens in one pass reads the weights once, so the extra tokens are nearly free.
- At large batch sizes the GPU is already busy, decode becomes more compute-bound, and the free capacity disappears. Speculative decoding helps most for low-latency, low-batch serving.
- Acceptance is high on predictable text (code, boilerplate, quoting the input) and lower on creative text. Medusa, EAGLE and n-gram prompt lookup are variants that avoid running a separate draft model.

<div class="take">
<div class="lab">Key takeaway</div>

speculative decoding lets a cheap draft propose tokens and the big model verify them all in one memory-bound pass. A careful acceptance rule keeps the output distribution identical, so you get 2-3x speed for free in quality.

</div>
