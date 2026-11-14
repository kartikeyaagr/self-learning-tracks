---
id: "16.03"
course: "inference-eng-course"
module: "16"
slug: "acceptance-rejection-draft-length"
title: "Acceptance, Rejection & Draft Length"
keyIdeas: ["Accept draft token x with probability min(1, p(x) / q(x)).", "On rejection, sample from normalized max(0, p - q).", "This is lossless: the output distribution equals the target's.", "Greedy case: accept while draft equals target argmax.", "Draft length k: longer gains more when acceptance is high, wastes more when low."]
question: "State the acceptance rule for speculative sampling and explain why it preserves the target model's output distribution."
answer: "For a draft token x with draft probability q(x) and target probability p(x), accept with probability min(1, p(x)/q(x)). If rejected, sample from the normalized residual max(0, p - q). Combining the two cases gives exactly p(x) for every token, so the final output is distributed as if sampled from the target alone. With greedy decoding, the rule reduces to accepting while the draft token equals the target's argmax."
sources: [{"title": "Accelerating Large Language Model Decoding with Speculative Sampling", "url": "https://arxiv.org/abs/2302.01318", "note": "Rejection sampling scheme."}, {"title": "Fast Inference from Transformers via Speculative Decoding", "url": "https://arxiv.org/abs/2211.17192", "note": "Proof of correctness."}]
tags: ["speculative"]
---

## The acceptance rule for sampling

With greedy decoding, acceptance is simple: accept a draft token if it equals the target's top token. With sampling, you need a rule that keeps the output distribution exactly equal to the target's. The standard rule uses the draft's probability q and the target's probability p for each drafted token.

For each draft position i, in order:

```
r = uniform(0, 1)
if r < min(1, p_i(x_i) / q_i(x_i)):
    accept x_i and continue to i + 1
else:
    sample y from normalize(max(0, p_i - q_i))
    output y and end the round
if every draft token was accepted:
    sample one more token from p_{k+1}
```

- If the target likes the token at least as much as the draft did (p ≥ q), it is always accepted.
- If the target likes it less, it is accepted with probability p ÷ q.
- On rejection, the replacement comes from the **residual distribution**: the part of p that q did not cover, renormalized.

## Why it is exact

For any token x at a given position, the probability that the round outputs x is:

P(x) = q(x) × min(1, p(x) ÷ q(x)) + P(reject) × residual(x)

The first term equals min(q(x), p(x)). The rejection probability is 1 − Σ min(q, p), and the residual is max(0, p − q) divided by that same quantity. So the second term equals max(0, p(x) − q(x)). Adding them gives min(q, p) + max(0, p − q) = p(x).

Every output token is therefore distributed exactly as if the target had sampled it. Quality is unchanged in distribution. Only speed changes.

<div class="ex">
<div class="lab">Worked example</div>

**the rule on one position**

The target gives "the" a probability of 0.5 and "a" 0.3. The draft gives "the" 0.7 and "a" 0.2. The draft proposed "the".

Acceptance probability: min(1, 0.5 ÷ 0.7) ≈ 0.71.

If rejected, the residual is max(0, p − q): "the" gets max(0, 0.5 − 0.7) = 0, "a" gets 0.1, and other tokens get their excess. The replacement is never "the" again.

Overall, "the" is output with probability 0.7 × 0.71 ≈ 0.5, exactly the target's probability.

</div>

## What acceptance depends on

- **Task.** Code, structured output and text that copies the input have high acceptance. Creative writing has low acceptance.
- **Temperature.** Higher temperature spreads both distributions, and they overlap less often token by token. Greedy decoding usually has the highest acceptance.
- **Position in the draft.** Acceptance usually falls for later positions, because the draft is conditioned on its own earlier guesses.
- **Draft quality.** A draft trained or distilled to match the target agrees more often.

## Choosing the draft length k

| Situation | Better k |
| --- | --- |
| High acceptance (code, repetitive text, greedy) | larger, 6 to 10 |
| Low acceptance (creative, high temperature) | smaller, 2 to 4 |
| Expensive draft | smaller |
| Large batch, little spare compute | smaller, or off |

~~~dg-raw
%%widget spec Lower the acceptance rate to 0.5 and see how quickly a long draft stops paying off.
a = 0.6
k = 6
c = 0.05
%%
~~~

## Adaptive draft length

Fixed k is a compromise. Some engines adjust it:

- **Per request**, from that request's recent acceptance rate.
- **Per step**, from the draft model's confidence: stop drafting when the draft's top probability falls below a threshold.
- **By load**, shrinking k as the batch grows.

## What to measure

- **Acceptance rate per position**: the fraction of rounds where position i was accepted, given that position i − 1 was.
- **Mean accepted length**: tokens per round, the number that drives speedup.
- **Draft time and verify time per round**, to check the cost side.

## Deep dive: proof that the sampled output is unchanged

Let the target distribution be ⟦m838⟧ and the draft distribution ⟦m839⟧. The draft proposes ⟦m840⟧. The verifier accepts with probability ⟦m841⟧. On rejection, it samples from the residual

⟦M836⟧

The probability of outputting ⟦m842⟧ is

⟦M837⟧

So every output token is distributed exactly as the target alone would produce. The acceptance probability is ⟦m843⟧: one minus the total variation distance between the two models.

~~~dg-raw
%%bars One position with four candidate tokens. The draft's proposal is kept with probability min(p, q); the rest comes from the residual (p − q)⁺.
unit = probability
stack = Accepted from the draft, Sampled from the residual
Token A (p 0.50, q 0.30) | 0.30+0.20
Token B (p 0.30, q 0.40) | 0.30+0
Token C (p 0.15, q 0.20) | 0.15+0
Token D (p 0.05, q 0.10) | 0.05+0
%%
~~~

Here ⟦m844⟧. Each bar's total is ⟦m845⟧, as the proof requires.

<div class="take">
<div class="lab">Key takeaway</div>

the min(1, p ÷ q) acceptance rule with residual resampling keeps outputs exactly distributed as the target's. Acceptance depends on task, temperature and position, so tune the draft length to the acceptance rate and to the spare compute.

</div>
