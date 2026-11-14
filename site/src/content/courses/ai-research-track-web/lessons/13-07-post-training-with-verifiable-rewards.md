---
id: "13.07"
course: "ai-research-track-web"
module: "13"
slug: "post-training-with-verifiable-rewards"
title: "Post-training with verifiable rewards"
keyIdeas: ["A modern post-training recipe runs supervised fine-tuning (SFT) on demonstrations, then preference tuning, then reinforcement learning with verifiable rewards (RLVR) on tasks a program can grade, such as maths answers and unit tests.", "SFT is next-token cross-entropy with a **loss mask**: only assistant tokens (including end-of-turn) contribute, never the system prompt, user turns or padding. A token mean gives long answers most of the weight (a 2,000-token answer gets 95% next to a 100-token one); a per-sequence mean weights conversations equally.", "RLVR replaces a learned reward model with a **verifier**. That removes one way to hack the reward, but verifiers have bugs (answer extraction, formatting, lenient matching, special-cased tests) that a policy will find, so training reward must be checked against a strict held-out grader and read samples.", "GRPO samples a group of completions per prompt and uses each one's reward minus the group mean, often divided by the group standard deviation, as its advantage. No value network is needed, but groups where every sample gets the same reward (probability $p^G + (1-p)^G$) contribute nothing, so keep prompts near a mixed pass rate.", "Evaluate with fixed decoding settings, the unbiased estimator pass@k $= 1 - \\binom{n-c}{k}/\\binom{n}{k}$ from $n \\ge k$ samples, and n-gram decontamination, because a contaminated benchmark measures memory, not skill."]
question: "In an RLVR run with GRPO you sample 64 prompts per step with 8 completions each. Logs show 20 prompts where all 8 completions are correct and 12 where all 8 are wrong. Meanwhile pass@1 on your held-out maths benchmark has risen from 40 to 70 percent in 200 steps, while two similar benchmarks moved by 2 points. What fraction of the batch is producing gradient, what would you change in sampling, and what do you suspect about the benchmark jump?"
answer: "Groups with identical rewards have zero advantage for every completion, so 32 of 64 prompts (256 of 512 completions, half the rollout compute) produce no policy gradient beyond any KL term. Filter prompts by pass rate: drop or down-sample those the model already always solves or always fails, and refill the batch until it holds enough mixed groups (sampling more prompts per step and discarding uniform groups, as DAPO-style dynamic sampling does, or keeping a running per-prompt pass-rate estimate to drive a curriculum). A pass rate near 50 percent is the most informative: at G = 8 almost no such groups are uniform, while at a 90 percent pass rate about 43 percent are. Revisit the always-failed prompts later, or raise G for them, since a rare success there gets a large advantage and is valuable signal. On the benchmark: a 30-point jump in 200 steps on one benchmark while similar benchmarks barely move suggests contamination or an evaluation or verifier artefact rather than skill. Check long n-gram overlap (13-grams, say) between the RL and SFT prompt sets and the benchmark questions and answers; check whether the evaluation's answer extraction, prompt template or decoding settings changed; compare the training reward with a strict grader to rule out a lenient verifier; and read a sample of newly-correct outputs to see whether they reason or recite. Report pass@1 averaged over several samples (for example n = 16 per problem with the unbiased estimator) at fixed temperature, top-p and maximum length, so the comparison is not sampling noise."
sources: [{"title": "Shao et al.: DeepSeekMath (2024)", "url": "https://arxiv.org/abs/2402.03300", "note": "Section 4: the GRPO objective, group-relative advantages and the outcome versus process supervision variants."}, {"title": "DeepSeek-AI: DeepSeek-R1 (2025)", "url": "https://arxiv.org/abs/2501.12948", "note": "R1-Zero: RL from a base model with rule-based accuracy and format rewards, and the multi-stage R1 recipe."}, {"title": "Lambert et al.: Tulu 3 (2024)", "url": "https://arxiv.org/abs/2411.15124", "note": "A fully open SFT, DPO and RLVR pipeline, including the verifiers and the benchmark decontamination procedure."}, {"title": "Rafailov et al.: Direct Preference Optimization (2023)", "url": "https://arxiv.org/abs/2305.18290", "note": "Section 4: deriving the DPO loss from the KL-regularised reward-maximisation objective."}]
practice: ["build-assistant-only-sft-loss-mask", "normalize-grpo-group-advantages", "grpo-group-relative-advantage", "compare-sequence-and-token-length-normalization", "dpo-closed-form-loss"]
tags: ["SFT", "RLVR", "GRPO"]
tools: ["PyTorch", "TRL", "vLLM"]
---

## What you will learn

- How post-training is staged (SFT, preference tuning, RL with verifiable rewards) and what each stage is for.
- How to build the SFT loss mask, and why the way you average over tokens decides which answers matter.
- What a **verifier** is and how a policy learns to exploit a buggy one.
- How GRPO turns a group of graded samples into advantages, and why some groups give no gradient at all.
- How to evaluate honestly: fixed decoding, unbiased pass@k, decontamination.

*Before you start:* next-token cross-entropy, the policy-gradient idea and PPO's clipped objective from the course's reinforcement-learning lessons, and the systems picture of 13.01 to 13.04.

Lesson 13.06 ended with a pretrained model: a strong predictor of web text that will happily answer a maths question with another maths question. **Post-training** turns it into an assistant that answers, follows instructions and reasons reliably. It uses a tiny fraction of pretraining compute but decides most of what users see. We follow the open recipes (Tulu 3, DeepSeek-R1) and focus on reinforcement learning against a program that checks the answer.

## The idea in one picture

Take one maths prompt: "What is the sum of the first 20 odd numbers?" The model samples eight answers. A short program, the **verifier**, extracts each final answer and compares it with the reference (400). It returns a **reward** of 1 for a match and 0 otherwise:

| Completion | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Final answer | 400 | 380 | 210 | 400 | 20 | 441 | 39 | 400
| Reward | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1

The group's average reward is 0.375. The learning signal is "how much better than the group was this answer?" Subtract the mean, divide by the spread, and each right answer gets **advantage** +1.29, each wrong one -0.77.

~~~dg-fig 13.07-group.svg
Left: the eight advantages for this prompt. Right ones are pushed up, wrong ones pushed down, and the pushes sum to zero. Right: dividing by the group's standard deviation makes a rare success on a hard prompt (few correct) count far more than a success on an easy one.
~~~

Training raises the probability of every token in the correct completions in proportion to +1.29 and lowers it in the wrong ones. Over thousands of prompts the model drifts toward whatever reasoning produces verifiably correct answers. That is **reinforcement learning with verifiable rewards (RLVR)**, and the advantage rule is **GRPO** (group relative policy optimisation).

There is no human labeller and no learned reward model, only a program. That is the method's strength and, as we will see, its weak point.

## Step 1: the post-training stack

Each stage starts from the previous checkpoint, with a far smaller learning rate than pretraining.

~~~dg-flow A typical open post-training pipeline.
Base model -(demonstrations)-> SFT -(chosen vs rejected pairs)-> Preference tuning (DPO or RM + PPO) -(prompts with checkable answers)-> *RLVR -> Evaluation and decontamination checks
~~~

- **Supervised fine-tuning (SFT)** imitates demonstrations: the chat format, when to stop, basic helpfulness.
- **Preference tuning** learns from pairs where one answer was preferred: tone, helpfulness, refusals.
- **RLVR** trains on tasks a program can check and is where most recent reasoning gains came from.

As in pretraining, the data matters more than a clever objective.

## Step 2: SFT and the loss mask

SFT data are conversations rendered with the model's **chat template** (the text format with role markers). The loss is next-token cross-entropy, except that a **loss mask** zeroes every position whose target token was not written by the assistant.

~~~dg-fig 13.07-mask.svg
Left: a rendered conversation. Only the assistant's tokens, including the end-of-turn token, have mask 1. Right: how much of the gradient a short and a long answer receive under the two common ways of averaging.
~~~

Training on user tokens teaches the model to *write* user turns; template markers carry nothing to learn; padding is wasted signal. Keep the end-of-turn token in the mask: that is how the model learns to stop. Position $t$ predicts token $t+1$, so the mask at position $t$ must follow the role of token $t+1$; an off-by-one is a classic bug.

Write $\ell_{s,t}$ for the cross-entropy at token $t$ of sequence $s$, $m_{s,t} \in \{0, 1\}$ for the mask and $T_s = \sum_t m_{s,t}$ for the number of trained tokens in sequence $s$. There are two common reductions over a batch of $S$ sequences:

$$L_{\text{token}} = \frac{\sum_s \sum_t m_{s,t}\,\ell_{s,t}}{\sum_s T_s}, \qquad L_{\text{seq}} = \frac{1}{S}\sum_s \frac{\sum_t m_{s,t}\,\ell_{s,t}}{T_s}.$$

The token mean weights every token equally, so long answers dominate; the sequence mean weights every conversation equally.

<div class="ex">
<div class="lab">Worked example</div>

**how much gradient does a long answer get?**

A batch holds two conversations with $T_1 = 100$ and $T_2 = 2000$ assistant tokens.

Token mean: the denominator is $100 + 2000 = 2100$. The short answer's share of the total weight is $100 / 2100 = 4.8\%$; the long answer's is $2000/2100 = 95.2\%$.

Sequence mean: each gets $50\%$.

Gradient accumulation trap: put each conversation in its own micro-batch with a token mean per micro-batch, and accumulating gives $50\%/50\%$, the sequence mean, not the token mean you wrote. Fix: divide by the token count of the *whole* effective batch.

</div>

When several short conversations are **packed** into one sequence to avoid padding, attention masks (or position resets) must stop one attending to another.

## Step 3: preference tuning, briefly

Preference data give a prompt $x$, a chosen response $y_w$ and a rejected one $y_l$. The classical route trains a Bradley-Terry **reward model** and optimises against it with PPO plus a KL penalty. **DPO** (direct preference optimisation) skips the reward model:

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\!\left(\beta\left[\log\frac{\pi(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log\frac{\pi(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right]\right),$$

where $\pi$ is the model being trained, $\pi_{\text{ref}}$ is the frozen SFT model, each $\log \pi(y \mid x)$ is the sum of token log-probabilities of the whole response, $\sigma$ is the logistic sigmoid and $\beta$ (often around 0.1) sets how far the policy may move from the reference.

Number check: chosen log-probability moves from -45 to -42 (+3), rejected from -48 to -50 (-2). The bracket is $5$, times $\beta = 0.1$ is $0.5$, and the loss is $-\log \sigma(0.5) = 0.474$, down from $\log 2 = 0.693$ at a tie.

Preference tuning improves style and refusals, but a learned reward model has blind spots: push hard against it and the policy finds responses it overrates (long, confident, flattering).

## Step 4: rewards from a program

For tasks with checkable outcomes, the reward can be computed by code:

- **Maths:** 1 if the extracted final answer equals the reference after normalisation, else 0.
- **Code:** the fraction of unit tests passed in a sandbox (or 1 only if all pass).
- **Instruction following:** whether constraints hold (word limit, format, keywords).

The Tulu 3 report named this setting RLVR. DeepSeek-R1-Zero showed its reach: from a base model with no SFT, RL with only rule-based accuracy and format rewards produced long reasoning chains and large maths gains.

A verifier cannot be swayed by confident prose. But **verifiers are code, and the policy will find their bugs:**

- Correct answers graded wrong because of formatting: $\tfrac{1}{2}$ versus 0.5, LaTeX, units, trailing full stops.
- Answers extracted from the wrong place, such as the first number in the text instead of the boxed final answer.
- Lenient matching that accepts an output listing several candidates ("the answer is 380, 400 or 441").
- Code that special-cases the visible tests, or crashes or hangs the grader.

## Step 5: group-relative advantages

PPO needs a **value network** as large as the policy. GRPO, introduced with DeepSeekMath, drops it. For each prompt $q$ it samples a **group** of $G$ completions $o_1, \dots, o_G$, scores them to get rewards $r_1, \dots, r_G$, and sets

$$A_i = \frac{r_i - \operatorname{mean}(r_1, \dots, r_G)}{\operatorname{std}(r_1, \dots, r_G) + \epsilon},$$

where $A_i$ is given to *every* token of completion $i$ and $\epsilon$ avoids dividing by zero. The group mean is a per-prompt **baseline**. The advantages enter a PPO-style clipped objective. With $\rho_{i,t} = \pi_\theta(o_{i,t} \mid q, o_{i,<t}) / \pi_{\text{old}}(o_{i,t} \mid q, o_{i,<t})$ (current over sampling policy),

$$J(\theta) = \frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\min\!\Big(\rho_{i,t}A_i,\ \operatorname{clip}(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon)\,A_i\Big) - \beta\, D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}}),$$

where $|o_i|$ is the completion length, $\varepsilon$ (often 0.2) the clipping range and $\beta$ weights a KL penalty toward a frozen reference. Three consequences follow.

**Uniform groups give nothing.** If all $G$ completions get the same reward (all solved or all failed), every $A_i$ is zero and the prompt contributes no policy gradient. For a prompt the model solves with probability $p$, that happens with probability $p^G + (1-p)^G$.

~~~dg-fig 13.07-dead.svg
Chance that a group carries no learning signal. Around a 50% pass rate almost every group is informative; near 0% or 100% most are wasted, and larger groups only partly help.
~~~

At $p = 0.5$ and $G = 8$ the waste is $2 \times 0.5^8 = 0.8\%$; at $p = 0.9$ it is $0.9^8 + 0.1^8 = 43\%$. So keep groups mixed: drop always-solved prompts and use a running per-prompt pass rate as a curriculum.

**Dividing by the standard deviation reweights prompts.** With binary rewards and pass rate $p$, a correct sample gets $A = \sqrt{(1-p)/p}$ and a wrong one $-\sqrt{p/(1-p)}$. At $p = 0.125$ (one success in eight) the success gets $+2.65$. Without the division it would get only $1 - p = 0.875$. It also boosts rare failures on nearly-solved prompts, and makes the step size depend on the prompt mix.

**Rollouts dominate cost.** Generation runs on an inference engine (such as vLLM) with weights copied from the trainer every step.

The figure below trains a toy **tabular policy**: 200 prompts with eight possible answers each (one correct), starting at 9.8% accuracy, 32 prompts per step, $G = 8$, same learning rate for three advantage rules.

~~~dg-fig 13.07-toyrun.svg
Left: accuracy over 300 steps for three advantage rules (toy simulation). Right: in the GRPO run, about half of all groups carry no signal at the start (hard prompts always fail) and about 70% by the end (easy prompts always succeed).
~~~

GRPO reached 75% by step 300, mean-only 47% and raw rewards 50%; part of that lead is simply that dividing by a small standard deviation is a bigger step. The real lesson is on the right: by the end about 70% of rollouts produce no gradient.

## Worked example by hand

<div class="ex">
<div class="lab">Worked example</div>

**one GRPO group and a pass@k estimate**

Rewards $[1, 0, 0, 1, 0, 0, 0, 1]$. Mean $= 3/8 = 0.375$.

Population variance $= p(1-p) = 0.375 \times 0.625 = 0.234$, so the standard deviation is $\sqrt{0.234} = 0.484$.

Correct completions: $A = (1 - 0.375)/0.484 = 0.625/0.484 = +1.29$. Wrong ones: $A = (0 - 0.375)/0.484 = -0.77$.

Check with one more digit: $3 \times 1.291 + 5 \times (-0.775) = 3.873 - 3.873 = 0$. Advantages from a mean baseline always sum to zero.

With the sample standard deviation (divide by $G - 1 = 7$): $0.518$, giving $+1.21$ and $-0.72$. Libraries differ.

Evaluation: draw $n = 16$ samples for a problem and find $c = 4$ correct. pass@1 $= 1 - \binom{12}{1}/\binom{16}{1} = 1 - 12/16 = 0.25$.

pass@4 $= 1 - \binom{12}{4}/\binom{16}{4} = 1 - 495/1820 = 1 - 0.272 = 0.728$.

pass@8 $= 1 - \binom{12}{8}/\binom{16}{8} = 1 - 495/12870 = 1 - 0.038 = 0.962$.

</div>

## Post-training pieces in code

The core formulas fit in a few lines of PyTorch (random logits stand in for a model).

```python
import torch, torch.nn.functional as F
from math import comb
torch.manual_seed(0)

# 1. SFT loss with an assistant-only mask
V = 50
ids  = torch.randint(0, V, (2, 12))
role = torch.tensor([[0,0,0,0,1,1,1,1,1,2,2,2],     # 0 user, 1 assistant, 2 pad
                     [0,0,1,1,1,1,1,1,1,1,1,1]])
logits = torch.randn(2, 12, V)                      # stand-in for model(ids).logits
labels = ids[:, 1:].clone()                         # position t predicts token t+1
mask = (role[:, 1:] == 1)                           # mask follows the LABEL token
labels[~mask] = -100                                # ignored by cross_entropy
tok_loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), labels.reshape(-1),
                           ignore_index=-100, reduction='none').view(2, 11)
per_token = tok_loss.sum() / mask.sum()
per_seq = (tok_loss.sum(1) / mask.sum(1)).mean()
print(mask.sum(1).tolist(), round(per_token.item(), 3), round(per_seq.item(), 3))
# [5, 10] 4.767 4.666

# 2. GRPO advantages for a batch of groups
def group_advantages(r, scale=True, eps=1e-6):
    a = r - r.mean(dim=1, keepdim=True)
    if scale:
        a = a / (r.std(dim=1, keepdim=True, unbiased=False) + eps)
    return a
r = torch.tensor([[1., 0, 0, 1, 0, 0, 0, 1],
                  [1., 1, 1, 1, 1, 1, 1, 1]])       # second group: all solved
print(group_advantages(r).round(decimals=2))
# [[ 1.29, -0.77, -0.77,  1.29, -0.77, -0.77, -0.77,  1.29],
#  [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00]]

# 3. Unbiased pass@k
def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)
print([round(pass_at_k(16, 4, k), 3) for k in (1, 4, 8)])   # [0.25, 0.728, 0.962]
```

The two SFT reductions already disagree (4.767 versus 4.666); in libraries such as TRL this is a configuration flag.

## Reading a run: is the model actually learning?

Training reward is the least trustworthy number on the dashboard. Also log:

- **Held-out accuracy with a strict grader.** If training reward rises and this does not, suspect the verifier.
- **Fraction of uniform groups.** Rising toward 100% means the prompt set is exhausted.
- **Response length, KL and entropy.** Sudden length jumps or entropy near zero are warnings.
- **Samples.** Read rewarded outputs regularly. Nothing replaces this.

Here is a verifier bug. Give the toy policy one shared "hedge" behaviour (list several candidate answers) that a lenient verifier accepts on every prompt; the fixed verifier rejects it.

~~~dg-fig 13.07-hack.svg
Toy simulation of reward hacking. With the buggy verifier, training reward climbs to 100% while true accuracy peaks at 44% at step 115 and then collapses toward zero. With the fixed verifier, reward and accuracy are the same curve and reach 75%.
~~~

Up to step 115 the runs match, so a short sanity run would miss the bug; then the exploit, which works on every prompt at once, outgrows any per-prompt skill. The training reward looks like a triumph; only a strict held-out grader or reading samples reveals it.

## Evaluation and contamination

Scores of sampled models depend on decoding, so fix temperature, top-p, maximum length and prompt template, and average over several samples.

**pass@k** is the probability that at least one of $k$ attempts is correct. Draw $n \ge k$ samples per problem, count $c$ correct, and use

$$\text{pass@}k = 1 - \binom{n-c}{k}\Big/\binom{n}{k},$$

one minus the chance that a random $k$ of your $n$ samples are all wrong, averaged over problems.

~~~dg-fig 13.07-passk.svg
Left: pass@k from 16 samples for three problems. Right: over 20,000 simulated problems with a true solve rate of 0.1, the unbiased estimator matches the truth while the plug-in formula 1 − (1 − c/n)^k is too low.
~~~

The tempting shortcut, plugging $\hat p = c/n$ into $1 - (1-\hat p)^k$, is biased low. In the simulation, the true pass@8 is 57.0%: the unbiased estimator averages 57.2%, the plug-in 49.3%. At $k = 16$ the gap grows to 81.5% versus 66.2%.

**Contamination** is the bigger threat: training data can contain benchmark questions and solutions, and then a score measures memory, not skill. Check long n-gram overlap between training sets and benchmarks (GPT-3 used 13-grams), remove matches as the Tulu 3 team did, and prefer benchmarks newer than the data cut-off. A jump on one benchmark with no movement on similar ones is a contamination smell.

## What goes wrong

| Problem | Symptom | Fix
| --- | --- | --- |
| Mask includes user turns | model writes "User:" lines | mask by role of the label token; unit-test the mask
| Mask drops end-of-turn token | answers never stop, run to max length | include the end-of-turn token in the mask
| Token-mean per micro-batch | loss and results change with micro-batch size | normalise by tokens in the whole effective batch
| Uniform groups | many zero advantages; reward plateaus | filter by pass rate, sample more prompts, curriculum
| Verifier too strict | correct answers scored 0, slow learning | normalise answers (fractions, LaTeX, units); audit false negatives
| Verifier too lenient | training reward near 100%, held-out flat or falling | strict extraction, hidden tests, sandbox with timeouts, read samples
| Length bias | wrong answers grow longer over training | sequence-level or constant normalisation (see below)
| Contaminated benchmark | one score jumps, siblings do not move | n-gram decontamination, newer benchmarks

## Going deeper

**Why a baseline is allowed.** The policy-gradient estimator is $\mathbb{E}[(r - b)\nabla_\theta \log \pi_\theta(o)]$. For any $b$ that does not depend on the sampled $o$, $\mathbb{E}[b\,\nabla_\theta \log \pi_\theta(o)] = b\,\nabla_\theta \sum_o \pi_\theta(o) = b\,\nabla_\theta 1 = 0$. A baseline changes the variance, not the expected gradient. The GRPO group mean includes the sample's own reward; the leave-one-out baseline (RLOO) uses the other $G - 1$ rewards, giving $r_i - \bar r_{-i} = \tfrac{G}{G-1}(r_i - \bar r)$, the same direction with a constant rescaling.

**Two biases in the original objective.** Later analysis (the "Dr. GRPO" paper, stated from memory) argued two normalisations are not neutral. Dividing by the standard deviation weights prompts by the model's uncertainty. Dividing by $|o_i|$ penalises a long wrong answer less *per token* than a short one, nudging the policy toward longer wrong answers. Their fix drops both. Other recipes, such as DAPO, average over all tokens in the batch and filter out uniform groups ("dynamic sampling").

**The KL term.** GRPO adds the KL penalty to the loss rather than to the reward, using the per-token estimator $\frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1$, which is always non-negative. Several reasoning recipes set $\beta = 0$.

**Off-policy staleness.** When generation runs ahead of training, the ratio $\rho_{i,t}$ and the clip correct for the older sampling policy. Engine and trainer log-probabilities can differ numerically, so many systems recompute them on the trainer.

## Key terms

| Term | Meaning
| --- | --- |
| SFT | Fine-tuning on demonstrations with cross-entropy on assistant tokens only
| DPO | Preference tuning with a closed-form loss on chosen versus rejected log-probability ratios
| Verifier | A program that grades an output (answer match, unit tests, format check)
| RLVR | Reinforcement learning whose reward comes from a verifier rather than a learned model
| GRPO | Policy optimisation with advantages from a group of samples per prompt, no value network
| Advantage | How much better a completion did than the baseline for its prompt
| pass@k | Probability that at least one of k samples is correct, estimated from n ≥ k samples
| Contamination | Benchmark items present in training data, inflating scores

<div class="take">
<div class="lab">Key takeaway</div>

mask SFT to assistant tokens and pick the loss reduction on purpose, use preference tuning for style and RLVR for skills a program can check, keep GRPO groups mixed so they carry signal, harden and audit the verifier as carefully as the model, and trust no score you have not measured with fixed decoding, unbiased pass@k and decontaminated benchmarks.

</div>
