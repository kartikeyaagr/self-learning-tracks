---
id: "08.01"
course: "ai-engineering-course"
module: "08"
slug: "reasoning-models-with-human-feedback"
title: "Reasoning models and human feedback"
keyIdeas: ["RLHF turns human comparisons into a reward model (Bradley-Terry loss), then optimises the LLM against it with PPO and a KL penalty that keeps it near the starting model.", "DPO reaches the same objective directly from preference pairs, with no reward model or RL loop, but inherits the same data biases.", "A learned reward is a proxy: push hard and the policy finds its blind spots (length, sycophancy, confident tone). That is reward hacking.", "Reasoning models (o1 onward, DeepSeek-R1, Qwen3, gpt-oss) are trained with RL on verifiable rewards, usually with GRPO, which scores each sample against the other samples for the same prompt instead of training a critic.", "Test-time scaling: accuracy rises with the tokens a model spends thinking (longer chains) and with the samples you draw (voting, best-of-n), with diminishing returns."]
question: "Compare RLHF with a learned reward model to RL with verifiable rewards trained with GRPO (as in DeepSeek-R1). What signal does each optimise, how is the update computed, and what can go wrong?"
answer: "RLHF optimises a reward model trained on human preference pairs with a Bradley-Terry loss; PPO then maximises that reward minus a KL penalty to the reference model, using a learned value model as the baseline. DPO folds the same objective into a classification loss on the pairs. It handles open-ended qualities like helpfulness, but the reward is a proxy, so heavy optimisation finds its blind spots: longer, flattering, overconfident answers. RL with verifiable rewards scores each sample with a program (answer matches, tests pass). GRPO samples a group of answers per prompt and uses each answer's reward minus the group mean, divided by the group standard deviation, as its advantage, so no critic is needed. The signal is objective and produced long, self-checking reasoning in R1, but it only covers checkable tasks, a weak checker can be gamed, groups where every sample gets the same reward teach nothing, and outcome-only rewards credit right answers reached by wrong reasoning."
rubric: ["States that RLHF optimises a learned proxy of human preference and that verifiable-reward RL optimises a programmatic check.", "Describes the RLHF update: Bradley-Terry reward model, then PPO with a KL penalty (or DPO as the direct alternative).", "Describes GRPO's group-relative advantage (reward minus group mean, over group std) and that it removes the value model.", "Names reward hacking for RLHF (length, sycophancy) and checker gaming or narrow coverage for verifiable rewards.", "Notes at least one GRPO-specific limit: zero-variance groups give no signal, or outcome rewards credit lucky answers (process rewards address this)."]
sources: [{"title": "Ouyang et al.: InstructGPT (2022)", "url": "https://arxiv.org/abs/2203.02155", "note": "The SFT, reward model, PPO pipeline."}, {"title": "Rafailov et al.: Direct Preference Optimization (2023)", "url": "https://arxiv.org/abs/2305.18290", "note": "Optimising the RLHF objective directly on preference pairs."}, {"title": "Shao et al.: DeepSeekMath (2024)", "url": "https://arxiv.org/abs/2402.03300", "note": "Introduces GRPO, PPO without a value model."}, {"title": "DeepSeek-AI: DeepSeek-R1 (2025)", "url": "https://arxiv.org/abs/2501.12948", "note": "Reasoning learned through RL with verifiable rewards; distilled small models."}]
practice: ["grpo-group-relative-advantage", "normalize-grpo-group-advantages", "bradley-terry-reward-model-loss", "dpo-closed-form-loss", "ppo-with-reference-policy-kl"]
tools: ["DeepSeek-R1", "Qwen3", "gpt-oss", "GPT-5"]
---

## From preferences to a reward

A pretrained and instruction-tuned model can answer, but "a good answer" is hard to write down as a loss. People find it much easier to compare two answers than to score one. RLHF turns those comparisons into a training signal. The InstructGPT pipeline has three stages:

~~~dg-flow The three stages of classic RLHF.
Base model -(SFT on demonstrations)-> SFT model -(humans rank outputs)-> *Reward model -(PPO + KL penalty)-> Aligned model
~~~

1. **SFT**: fine-tune on human-written ideal responses.
2. **Reward model**: labellers see two answers to the same prompt and pick the better one. A model $r(x, y)$ learns to score the chosen answer $y_w$ above the rejected one $y_l$ with the Bradley-Terry loss:

$$
\mathcal{L}_{RM} = -\log \sigma\big(r(x, y_w) - r(x, y_l)\big)
$$

3. **RL**: optimise the policy $\pi$ to maximise the reward while staying close to the SFT model $\pi_{ref}$:

$$
\max_\pi \; \mathbb{E}\big[r(x, y)\big] - \beta \, \mathrm{KL}\big(\pi \,\|\, \pi_{ref}\big)
$$

PPO is the usual optimiser. It needs a second learned network, a value model (critic), to estimate how good each state is, so classic RLHF trains and holds four models in memory: policy, reference, reward and value.

The reward loss only cares about differences. If the reward model scores the chosen answer 2.0 and the rejected one 0.5, $\sigma(1.5) = 0.82$ and the loss is 0.20. If it has them backwards, $\sigma(-1.5) = 0.18$ and the loss is 1.71. Adding 10 to every score changes nothing, so the reward scale is arbitrary.

## Reward hacking

A reward model is a learned approximation of human judgement trained on limited data. Optimise hard against it and the policy finds its blind spots. Gao, Schulman and Hilton (2022) measured this: as the policy moves further from where it started (measured by KL), the proxy reward keeps climbing while a held-out "gold" reward rises, peaks and falls.

~~~dg-raw
%%widget rewardhack Proxy reward keeps rising while true quality peaks and falls as RL pushes harder.
~~~

The familiar symptoms: answers get longer because length correlated with preference; models agree with the user's stated view (sycophancy) because labellers liked being agreed with; tone gets more confident without getting more accurate. The KL leash, reward model ensembles and fresh preference data limit this but never remove it.

## DPO: the same goal without a reward model

Rafailov et al. (2023) showed the KL-regularised objective above has a closed-form optimum, which lets you train directly on preference pairs with a classification-style loss:

$$
\mathcal{L}_{DPO} = -\log \sigma\Big(\beta \Big[\log \tfrac{\pi(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \log \tfrac{\pi(y_l \mid x)}{\pi_{ref}(y_l \mid x)}\Big]\Big)
$$

No reward model, no sampling during training, no critic. It is simpler and more stable than PPO, which is why it became the default preference step for open models. It learns from the same pairs, so it inherits their biases (longer answers win), and because it never samples from the current policy it can drift on inputs unlike its training pairs. For when to use DPO or reinforcement fine-tuning on your own product, see [Preference tuning for products](lesson:aieng:07.09).

## Verifiable rewards and reasoning models

For maths and code there is a better signal than an opinion: check the answer. DeepSeek-R1's reward is a rule. It gives 1 if the final answer matches the reference (or the code passes its tests), with a small extra reward for putting the reasoning in the expected format.

| Aspect | RLHF with reward model | RL with verifiable rewards |
| --- | --- | --- |
| Signal | Learned model of human preference | Programmatic check (answer match, tests) |
| Works for | Helpfulness, tone, safety, open-ended tasks | Maths, code, logic, anything checkable |
| Main risk | Hacking the reward model | Gaming a weak checker; narrow skills |
| What emerges | Polite, helpful style | Long reasoning, self-verification, backtracking |

DeepSeek-R1-Zero, trained with RL alone from a base model, learned on its own to write longer reasoning, re-check its work and say things like "wait, let me reconsider". Its raw output mixed languages and was hard to read, so the released R1 uses a pipeline: a small cold-start SFT set of long, readable reasoning traces, reasoning RL, about 800k rejection-sampled SFT examples, then a final RL stage with both rule rewards and preference reward models for general helpfulness. The same 800k examples were used to distil R1 into dense Qwen and Llama models from 1.5B to 70B parameters ([Data for fine-tuning: synthetic data and distillation](lesson:aieng:07.08)). Later open reasoning models such as Qwen3 and gpt-oss follow the same broad recipe of SFT on reasoning traces plus RL on checkable tasks.

## GRPO: RL without a critic

Group Relative Policy Optimization, introduced in DeepSeekMath and used for R1, drops PPO's value model. For each prompt it samples a group of $G$ answers, scores each, and uses the group itself as the baseline:

$$
\hat{A}_i = \frac{r_i - \mathrm{mean}(r_1, \dots, r_G)}{\mathrm{std}(r_1, \dots, r_G)}
$$

Every token of answer $i$ gets advantage $\hat{A}_i$, plugged into a PPO-style clipped objective, with the KL term to the reference model added directly to the loss. Answers that beat their siblings are made more likely; answers that lose are made less likely. Removing the critic saves a model the size of the policy, which matters at hundreds of billions of parameters.

<div class="ex">
<div class="lab">Worked example</div>

**group advantages for one maths prompt**

Sample $G = 4$ answers. Two reach the right final answer, two do not: rewards $[1, 0, 0, 1]$. The mean is 0.5 and the (population) standard deviation is 0.5, so the advantages are $[+1, -1, -1, +1]$. Each correct answer's tokens are pushed up; each wrong answer's tokens are pushed down.

Now a prompt that is too hard: rewards $[0, 0, 0, 0]$. The mean is 0 and every advantage is 0 (implementations add a small epsilon to the denominator to avoid dividing by zero). The prompt contributes no gradient at all. The same happens for a prompt that is too easy, $[1, 1, 1, 1]$. Only prompts the model sometimes solves teach it anything, which is why later recipes such as DAPO filter out all-correct and all-wrong groups and keep sampling until the batch is informative.

At the scale of a full run the signal is strong: R1-Zero's AIME 2024 pass@1 rose from 15.6% to 71.0% over training, with long reasoning and self-checking emerging without being taught.

</div>

Two caveats. Dividing by the group standard deviation weights low-variance prompts heavily, and averaging the loss over each answer's length penalises a long wrong answer less per token than a short one, which nudges wrong answers to grow. Follow-up work (Dr. GRPO, among others) changes both normalisations. And an outcome reward credits a right answer reached by broken reasoning. Process reward models, which score each step, address that; [Diagnose four reasoning failures](lesson:aieng:08.05) shows how step-level verification is used at inference.

## Test-time scaling

RL on verifiable rewards produced a second scaling axis. OpenAI's o1 report showed accuracy improving smoothly as test-time compute grew (plotted on a log scale), and R1-Zero's average response length grew from hundreds to thousands of tokens during training. Two ways to spend that compute:

- **Sequential**: one longer chain. Reasoning models do this by default, and APIs expose it as a reasoning effort or thinking budget. The s1 paper showed the effect directly: forcing a small fine-tuned model to keep thinking (appending "Wait" when it tried to stop) raised its competition-maths scores.
- **Parallel**: many chains, then pick. Majority voting ([Chain of thought](lesson:aieng:08.02)) or best-of-n with a verifier ([Tree of thought](lesson:aieng:08.04)).

Both have diminishing returns, and past a point longer thinking loops and second-guesses. How to choose a budget, what thinking tokens cost and when a reasoning model is the wrong tool are in [Using reasoning models in apps](lesson:aieng:08.06).

## Where this breaks

- **Unverifiable tasks.** Verifiable rewards do not exist for "write a kind reply". Those still need preference data, with all its biases.
- **Weak checkers.** A unit test that only checks the output type, or an answer matcher that accepts "42" anywhere in the text, will be found and exploited.
- **Narrow transfer.** Gains on maths and code transfer only partly to other reasoning, and RL can reduce output diversity.
- **Preference data is not neutral.** Whoever labels the pairs defines "better", and DPO and RLHF faithfully learn their quirks.

<div class="take">
<div class="lab">Key takeaway</div>

RLHF and DPO optimise a learned model of human preference and risk reward hacking. Reasoning models add RL on verifiable rewards, typically with GRPO, which ranks each sample against its siblings instead of training a critic. The result is a model whose accuracy scales with how long it thinks and how many samples you draw.

</div>
