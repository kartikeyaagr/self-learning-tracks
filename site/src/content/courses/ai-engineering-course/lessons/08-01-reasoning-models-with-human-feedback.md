---
id: "08.01"
course: "ai-engineering-course"
module: "08"
slug: "reasoning-models-with-human-feedback"
title: "Reasoning models with human feedback"
keyIdeas: ["RLHF: collect human comparisons of responses, train a reward model on them, then optimise the LLM against the reward model (PPO), with a KL penalty to stay near the original.", "DPO does the same job directly from preference pairs, with no separate reward model or RL loop.", "Reward hacking: the policy finds outputs the reward model scores highly but humans don't actually prefer (long, flattering, confident answers).", "Reasoning models (o1, DeepSeek-R1) are trained with RL on tasks with checkable answers: maths results, passing unit tests. The reward is verifiable, not a learned opinion.", "DeepSeek-R1-Zero showed long chains of thought, self-checking and backtracking emerging from RL alone on a base model.", "Outcome reward models score the final answer; process reward models score each step, which Lightman et al. found gives better supervision for maths."]
question: "Compare RLHF with a learned reward model to RL with verifiable rewards (as in DeepSeek-R1). What signal does each optimise, and what can go wrong?"
answer: "RLHF optimises a learned reward model trained on human preference comparisons (via PPO with a KL penalty, or directly via DPO); it handles open-ended qualities like helpfulness but can be reward-hacked (verbosity, sycophancy) because the reward model is only a proxy. RL with verifiable rewards optimises a programmatic check, such as the correct final maths answer or passing tests; the signal is objective and drove emergent long reasoning in R1, but it only covers checkable tasks and can be gamed if the checker is weak (e.g. right answer from wrong reasoning, which process rewards address)."
sources: [{"title": "Ouyang et al.: InstructGPT (2022)", "url": "https://arxiv.org/abs/2203.02155", "note": "The SFT, reward model, PPO pipeline."}, {"title": "Hugging Face: Illustrating RLHF", "url": "https://huggingface.co/blog/rlhf", "note": "Diagrams of each RLHF stage."}, {"title": "DeepSeek-AI: DeepSeek-R1 (2025)", "url": "https://arxiv.org/abs/2501.12948", "note": "Reasoning learned through RL with verifiable rewards."}, {"title": "Lightman et al.: Let's Verify Step by Step (2023)", "url": "https://arxiv.org/abs/2305.20050", "note": "Process reward models that grade each reasoning step."}]
---

## From preferences to a reward

A pretrained and fine-tuned model can answer, but "good answer" is hard to write down as a loss. RLHF turns human judgement into a training signal. The InstructGPT pipeline, drawn step by step in Hugging Face's RLHF explainer:

~~~dg-flow The three stages of classic RLHF.
Base model -(SFT on demonstrations)-> SFT model -(humans rank outputs)-> *Reward model -(PPO + KL penalty)-> Aligned model
~~~

1. **SFT**: fine-tune on human-written ideal responses.
2. **Reward model**: humans compare pairs of model outputs; a model learns to predict which one they prefer, outputting a scalar score.
3. **RL**: optimise the LLM to maximise the reward model's score using PPO, with a KL penalty that keeps it close to the SFT model so it doesn't drift into nonsense that happens to score well.

**DPO** (Direct Preference Optimization) reaches a similar result with a simple classification-style loss on the preference pairs, with no reward model and no RL loop, which is why it became popular for open models.

## Reward hacking

A learned reward model is only an approximation of human preference. Optimise hard against it and the policy finds its blind spots: longer answers, more confident tone, agreeing with the user. The KL penalty, reward model ensembles, and fresh human data limit this but never remove it.

## Verifiable rewards and reasoning models

For maths and code there's a better signal: check the answer. DeepSeek-R1's paper describes training with RL where the reward is simply whether the final answer is correct (and in the right format), or whether code passes tests.

| Aspect | RLHF with reward model | RL with verifiable rewards |
| --- | --- | --- |
| Signal | Learned model of human preference | Programmatic check (answer match, tests) |
| Works for | Helpfulness, tone, safety, open-ended tasks | Maths, code, logic, anything checkable |
| Main risk | Reward hacking the reward model | Gaming the checker; narrow skills |
| What emerges | Polite, helpful style | Long reasoning, self-verification, backtracking |

DeepSeek-R1-Zero, trained with RL only from a base model, learned on its own to write longer reasoning, re-check its work and say things like "wait, let me reconsider". The final R1 adds a small SFT warm-up for readability and a later preference stage for general helpfulness.

## Outcome vs process rewards

An **outcome reward** only checks the final answer, so a lucky guess after bad reasoning gets full marks. A **process reward model** scores each step. Lightman et al.'s "Let's Verify Step by Step" trained one on 800k step-level human labels and found it selected correct maths solutions much more reliably than an outcome reward model.

## How the reward model is trained

Labelers see two answers to the same prompt and pick the better one. The reward model r(prompt, answer) is trained so the chosen answer scores higher, using the Bradley-Terry loss:

```
loss = -log sigmoid( r(prompt, chosen) - r(prompt, rejected) )
```

If the reward model gives the chosen answer 2.0 and the rejected one 0.5, the difference is 1.5, sigmoid(1.5) = 0.82, and the loss is 0.20. If it had them backwards (0.5 versus 2.0), sigmoid(−1.5) = 0.18 and the loss is 1.71. Only differences matter, so the reward scale is arbitrary.

The policy is then trained with RL (usually PPO) to maximise:

```
reward(prompt, answer) - beta * KL( policy || reference model )
```

The KL term is the leash. It keeps the policy near the supervised model it started from, so it can't drift into strange text that happens to score well.

## Watch over-optimisation happen

The reward model is a learned approximation of human judgement, trained on limited data. Push the policy hard against it and the policy finds the reward model's blind spots. Gao, Schulman and Hilton (2022) measured this: as optimisation moves the policy further from the start (measured by KL), the proxy reward keeps climbing while the true "gold" reward rises, peaks and falls.

~~~dg-raw
%%widget rewardhack Proxy reward keeps rising while true quality peaks and falls as RL pushes harder.
~~~

Familiar real symptoms: answers get longer because length correlated with preference; models agree with the user's stated opinion (sycophancy) because labelers liked being agreed with; confident tone without more accuracy.

## DPO: the same goal without a reward model

Direct Preference Optimization (Rafailov et al., 2023) showed the RLHF objective can be optimised directly on preference pairs with a classification-style loss, skipping the separate reward model and the RL loop. It is simpler and more stable, and became a common choice for open models. It still learns from the same preference data, so it inherits the same biases.

<div class="ex">
<div class="lab">Worked example</div>

**why verifiable rewards changed reasoning training**

For a maths problem with a known answer, the reward is a rule: 1 if the final answer matches, 0 if not. There is nothing to hack except actually getting the answer right (or exploiting a bug in the checker).

DeepSeek-R1-Zero was trained this way with GRPO, and its AIME 2024 pass@1 rose from 15.6% to 71.0% over training, with long reasoning and self-checking emerging without being taught.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

RLHF optimises a learned model of human preference and risks reward hacking. Reasoning models add RL on verifiable rewards, where checking the answer is the signal, and long careful reasoning emerges from it.

</div>
