---
id: "16.06"
course: "inference-eng-course"
module: "16"
slug: "medusa-eagle-multi-token-prediction"
title: "Medusa, EAGLE & Multi-Token Prediction"
keyIdeas: ["Learned drafters use the target's own hidden states instead of a separate model.", "Medusa: extra heads predict tokens at offsets +1..+k; verify a tree of candidates.", "EAGLE: a light autoregressive head over target features; higher acceptance.", "Multi-token prediction (MTP): the model is trained with heads for future tokens, usable as drafters.", "Tree attention verifies several candidate continuations in one pass."]
question: "How do Medusa heads and EAGLE differ in what they use to predict future tokens?"
answer: "Medusa adds several extra decoding heads on the target model's last hidden state, each predicting the token at a further offset (t+1, t+2, ...), and verifies candidate combinations with tree attention. EAGLE trains a small autoregressive head that operates at the feature level, taking the target's hidden states (plus the sampled tokens) to predict the next features and tokens, which gives higher acceptance than independent heads."
sources: [{"title": "Medusa", "url": "https://arxiv.org/abs/2401.10774", "note": "Multiple decoding heads with tree verification."}, {"title": "EAGLE", "url": "https://arxiv.org/abs/2401.15077", "note": "Feature-level speculative drafting."}, {"title": "Better & Faster Large Language Models via Multi-token Prediction", "url": "https://arxiv.org/abs/2404.19737", "note": "Training models to predict several tokens."}]
tags: ["speculative"]
---

## Why learned drafters

A separate draft model has real costs: its own weights and KV cache in GPU memory, a sequential drafting loop, and the need for a model with the same tokenizer. A different approach attaches small prediction modules to the target model itself. They read the target's own hidden states, which already contain a lot of information about what comes next, so a tiny module can guess well.

## Medusa

Medusa adds k small heads on top of the target's last hidden state.

- Head i predicts the token at position t + i, directly from the hidden state at position t.
- The top candidates from each head are combined into a **tree** of possible continuations: for example the top 3 from head 1, each followed by the top 2 from head 2.
- The target verifies the whole tree in one pass, using a **tree attention mask** so each candidate path attends only to its own ancestors.
- The longest path the target accepts is kept.

Only the heads are trained, with the target model frozen, so training is cheap. The weakness: each head predicts its position without knowing what the earlier heads guessed, so acceptance drops quickly for later positions.

## EAGLE

EAGLE predicts at the level of features instead of tokens.

- A small module, about one transformer layer, takes the target's hidden feature at the current position and the token just sampled, and predicts the next hidden feature.
- The target's own LM head turns the predicted feature into a token distribution.
- The module runs autoregressively for k steps, so each guess is conditioned on the earlier guesses.

Predicting the next feature is easier than predicting the next token, because features vary more smoothly. EAGLE reaches higher acceptance than Medusa at a similar cost. Later versions build **dynamic draft trees**, expanding branches where the drafter is confident and pruning them where it is not, and train the drafter on features from several layers.

## Multi-token prediction

Some models are trained from the start with extra heads that predict tokens several steps ahead. The training objective improves the main model's representations, and at inference the extra heads serve as a built-in drafter. DeepSeek V3, for example, was trained with a multi-token prediction module, and its acceptance as a drafter is reported to be high for the next token.

## Comparing the options

| Method | Extra model | Training needed | Typical acceptance | Draft cost |
| --- | --- | --- | --- | --- |
| Separate draft model | yes | none if a small sibling exists | medium | one small model pass per token |
| N-gram lookup | no | none | task dependent | almost free |
| Medusa | heads | heads only | medium, falling fast with position | one pass for all heads |
| EAGLE | one small layer | the layer only | high | one small layer per token |
| Multi-token prediction | built in | part of pre-training | high | small |

~~~dg-raw
%%widget spec Compare a cheap drafter with modest acceptance (c 0.02, a 0.7) against a costlier one with high acceptance (c 0.1, a 0.85).
a = 0.85
k = 5
c = 0.03
%%
~~~

<div class="ex">
<div class="lab">Worked example</div>

**choosing a drafter**

A team serves a 70B chat model at low concurrency and wants lower TPOT.

A 1B sibling draft: acceptance about 0.72, draft cost 10% of a target pass. Best speedup about 2.0 times at k = 4.

An EAGLE head trained on their traffic: acceptance about 0.82, draft cost about 3%. Best speedup about 2.8 times at k = 5.

The EAGLE head also used less memory than the 1B model and its KV cache. They chose EAGLE and kept n-gram drafting for the code-editing endpoint.

</div>

## Practical notes

- Learned drafters are trained for one specific target model. A new version of the target usually needs a new drafter.
- Training on data like your traffic, or on the target's own outputs, raises acceptance.
- Tree verification processes more positions per round than a single chain, which costs more compute. It pays at low load and needs the same load-aware backoff as other methods.

## Deep dive: expected length with a tree of candidates

Learned drafters (Medusa, EAGLE) propose several candidates per position and verify a tree in one pass. If ⟦m856⟧ is the probability that the true token is among the candidates kept at depth ⟦m857⟧, given that all earlier depths matched:

⟦M855⟧

with ⟦m858⟧ the branching at depth ⟦m859⟧.

<div class="ex">
<div class="lab">Worked example</div>

**a small tree**

Branching (4, 2, 2, 1) gives ⟦m860⟧ nodes. With top-⟦m861⟧ recalls of 0.8, 0.7, 0.6 and 0.5:

⟦m862⟧ tokens per step.

At batch 1, 44 extra positions sit under the ridge and cost almost nothing. At batch 32 they become ⟦m863⟧ tokens per step, far above it. Tree width must shrink as batch grows.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

learned drafters reuse the target's hidden states to guess well at small cost. EAGLE-style and multi-token prediction drafters usually reach the highest acceptance per unit of cost, while Medusa is simpler but loses acceptance quickly with position.

</div>
