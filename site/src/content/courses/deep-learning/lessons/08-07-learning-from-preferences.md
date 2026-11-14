---
id: "08.07"
course: "deep-learning"
module: "08"
slug: "learning-from-preferences"
title: "Learning from preferences"
keyIdeas: ["When a reward is hard to write down, collect pairwise human judgements instead: people agree far more on \"which of these two is better\" than on absolute scores.", "The **Bradley-Terry model** says $P(y_w \\succ y_l) = \\sigma(r(y_w) - r(y_l))$. A reward model is trained with the loss $-\\log\\sigma(r_w - r_l)$, and only reward differences are identified, so absolute values need normalising.", "Label noise caps reward-model accuracy (often only 65 to 75 percent on held-out pairs); once you reach annotator agreement, more data of the same kind does not help.", "RLHF maximises $\\mathbb E[r_\\phi(x,y)] - \\beta\\,\\mathrm{KL}(\\pi \\| \\pi_\\text{ref})$, usually with PPO; the KL keeps the policy where the reward model is trustworthy, and without it the policy hacks the reward (proxy rises, true quality falls).", "DPO uses the closed-form optimum $\\pi^* \\propto \\pi_\\text{ref}\\,e^{r/\\beta}$ to turn that objective into a classification loss $-\\log\\sigma(\\beta[\\Delta_w - \\Delta_l])$ with $\\Delta = \\log\\pi - \\log\\pi_\\text{ref}$: no reward model, no sampling.", "Watch KL, answer length and the chosen and rejected log-probabilities separately; $\\beta$ is the main control against reward hacking and likelihood collapse."]
question: "After DPO training, your model's win rate against the reference improves, but you notice the log-probability of the chosen answers on the training set went down, not up. Is this a bug? Explain using the DPO loss, and say what risk it signals and what you would monitor or change."
answer: "It is not necessarily a bug. The DPO loss is $-\\log\\sigma(m)$ with margin $m = \\beta[(\\log\\pi(y_w) - \\log\\pi_\\text{ref}(y_w)) - (\\log\\pi(y_l) - \\log\\pi_\\text{ref}(y_l))]$. It depends only on the difference between the chosen and rejected log-ratios, not on either one alone, so the loss falls whenever the rejected answer's log-probability drops faster than the chosen one's, even if both drop. In practice the chosen and rejected answers for a prompt are usually similar and share most of their tokens, so a gradient step that pushes $y_l$ down also pushes $y_w$ down; the loss is satisfied as long as $y_l$ falls more. Where does the probability go? To answers that are in neither column of the dataset. The improved win rate says that, so far, much of that mass went somewhere good. The risk is that with a finite dataset DPO only constrains the answers it has seen, so mass can keep flowing to off-distribution, overly long or degenerate outputs that no pair ever penalised: a form of over-optimisation that the implicit KL does not prevent. What to do: monitor chosen and rejected log-probabilities separately, the KL from the reference on fresh prompts, answer length, and a held-out human or model-judged win rate rather than only the training loss. Mitigations include a larger $\\beta$ (shorter leash), fewer epochs or early stopping on the held-out judgement, adding a supervised negative log-likelihood term on the chosen answers so their probability cannot collapse, and on-policy or iterative variants that regenerate pairs from the current policy so the loss sees the outputs the model now actually produces."
sources: [{"title": "Christiano et al.: Deep Reinforcement Learning from Human Preferences", "url": "https://arxiv.org/abs/1706.03741", "note": "The original reward-model-from-comparisons method on Atari and MuJoCo."}, {"title": "Ouyang et al.: Training Language Models to Follow Instructions with Human Feedback", "url": "https://arxiv.org/abs/2203.02155", "note": "InstructGPT: SFT, reward modelling and PPO with a per-token KL penalty."}, {"title": "Rafailov et al.: Direct Preference Optimization", "url": "https://arxiv.org/abs/2305.18290", "note": "The closed-form optimum and the derivation of the DPO loss (sections 4 and 5)."}, {"title": "Nathan Lambert: Reinforcement Learning from Human Feedback (book)", "url": "https://rlhfbook.com/", "note": "Free book covering reward models, regularisation, PPO and direct alignment methods in practice."}]
practice: ["bradley-terry-reward-model-loss", "dpo-closed-form-loss", "canonicalize-pairwise-preference-data", "ppo-with-reference-policy-kl", "score-kl-regularized-policy-batch"]
tags: ["RLHF", "Bradley-Terry", "DPO"]
tools: ["PyTorch"]
---

## What you will learn

- Why "which of these two is better?" is often the only reward signal you can get.
- How the **Bradley-Terry model** turns pairwise votes into a reward, and how a reward model is trained with one line of loss.
- How RLHF optimises a policy against that reward while a KL penalty keeps it where the reward can be trusted.
- How DPO reaches the same objective with a plain classification loss, without a reward model or sampling.
- The real failure modes: reward hacking, length bias, label noise, and DPO lowering the probability of the answers it prefers.

*Before you start:* you need the RL objective and the idea of a policy from lesson 08.01, policy gradients and PPO-style actor-critic from lesson 08.06, logistic regression and the sigmoid (Machine Learning lesson 02.03), and KL divergence as a measure of how far one distribution is from another.

## Every method so far needed a reward

Lessons 08.01 to 08.06 assumed the environment hands out a reward $r$ after each action; actor-critic with PPO (08.06) was the most capable tool for maximising it. But often nobody can write the reward down. What is the reward for "a helpful, honest answer" or "a backflip that looks natural"?

People are also poor at absolute scores (one annotator's 7 is another's 5), but they are good at *comparing*: shown two summaries, most agree quickly on the better one. **Learning from preferences** collects such comparisons, fits a reward model that explains them, and optimises a policy against it.

Christiano et al. (2017) did this for Atari and simulated robots with feedback on under 1 percent of interactions; InstructGPT (Ouyang et al., 2022) applied it to language models, making **RLHF** (reinforcement learning from human feedback) the standard last stage of training chat models. The AI Research add-on covers the full post-training pipeline; this lesson covers the objectives.

~~~dg-flow The RLHF pipeline: people compare, a reward model generalises their judgements, RL optimises against it while staying near the reference.
Prompts -(sample two answers)-> Human comparisons -(Bradley-Terry fit)-> *Reward model -(PPO with KL penalty)-> Tuned policy
~~~

## The idea in one picture

A model writes four summaries of the same article, called A, B, C and D. Annotators see random pairs and pick the better one. Each of the six pairs is judged 20 times, 120 votes in all:

| Pair | A vs B | A vs C | A vs D | B vs C | B vs D | C vs D
| --- | --- | --- | --- | --- | --- | --- |
| First one wins | 14 | 16 | 18 | 12 | 15 | 13
| Win rate | 0.70 | 0.80 | 0.90 | 0.60 | 0.75 | 0.65

The votes are noisy (A still loses 2 of 20 to D) but ordered. We want one number per summary, its **reward**, whose gaps predict the win rates. The Bradley-Terry fit (step 1) gives A = 2.04, B = 1.12, C = 0.67, D = 0 (pinned).

~~~dg-fig 08.07-table.svg
Left: one reward per summary, fitted to the 120 votes. Right: the six observed win rates next to the sigmoid of the reward gaps. Four numbers explain six win rates almost exactly.
~~~

Read it like this: A beats D in 90 percent of votes and the model predicts $\sigma(2.04 - 0) = 0.885$; B beats C in 60 percent and the model predicts $\sigma(1.12 - 0.67) = 0.61$. A **reward model** does the same at scale: a neural network that scores any prompt and answer, including answers nobody voted on.

## Step 1: the Bradley-Terry model

The **Bradley-Terry model** (1952, originally for ranking sports teams and taste tests) says each item has a hidden score, and the chance that one item beats another depends only on the difference of their scores, through the sigmoid $\sigma(z) = 1/(1+e^{-z})$:

$$P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big)$$

Here $x$ is the prompt, $y_w$ the answer that won (the **chosen** answer), $y_l$ the answer that lost (the **rejected** answer), $y_w \succ y_l$ reads "$y_w$ is preferred to $y_l$", and $r(x, y)$ is the reward of answer $y$ for prompt $x$.

Number check: a gap of 0 gives 0.5 (a coin flip), a gap of 1 gives 0.73, a gap of 3 gives 0.95. It is logistic regression on the score difference, and the same curve as chess Elo ratings up to a change of scale.

~~~dg-fig 08.07-bt.svg
Left: the win probability is a sigmoid of the reward gap. Right: the reward-model loss and its gradient weight. Wrongly ranked pairs (negative gap) get a large push; pairs already far apart get almost none.
~~~

Only *differences* of rewards appear: add 100 to every reward and no probability changes. The reward is identified only up to a constant per prompt, so absolute values mean nothing until normalised (say, subtract the mean reward on a reference set). That is why D was pinned at zero.

## Step 2: train a reward model on comparisons

For language models the reward model $r_\phi$ (parameters $\phi$) is usually the fine-tuned network with its word-prediction layer replaced by one output number, read at the answer's last token. It is trained by maximum likelihood on triples $(x, y_w, y_l)$:

$$\mathcal L_{\text{RM}}(\phi) = -\,\mathbb E_{(x,\,y_w,\,y_l)}\Big[\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\Big]$$

In words: score both answers, take the gap, and pay $-\log \sigma(\text{gap})$. Its derivative with respect to the gap is $-(1 - \sigma(\text{gap}))$, so each pair pushes the scores apart with weight $1 - \sigma(\text{gap})$: large when ranked wrongly, tiny when already right.

<div class="ex">
<div class="lab">Worked example</div>

**one reward-model loss, by hand**

For one prompt, the reward model scores the chosen answer 1.2 and the rejected answer 0.4. The gap is $1.2 - 0.4 = 0.8$.

Probability the model assigns to the human's choice: $\sigma(0.8) = 1/(1 + e^{-0.8}) = 1/(1 + 0.449) = 0.690$.

Loss: $-\ln 0.690 = 0.371$ nats.

Gradient weight: $1 - 0.690 = 0.310$. Gradient descent raises the chosen score and lowers the rejected one, each in proportion to 0.31.

Compare a pair already separated by 5: $\sigma(5) = 0.9933$, weight $1 - 0.9933 = 0.0067$, about 46 times smaller. Easy pairs barely contribute; the model's effort goes to pairs it gets wrong or nearly wrong.

</div>

The four-summary table was fitted with exactly this loss, one free parameter per summary (average loss 0.55 nats per vote).

## Step 3: know the ceiling set by noisy labels

Annotators disagree. If they agree only 75 percent of the time, no reward model can predict a vote much better than that. Chat reward models typically reach around 65 to 75 percent held-out pair accuracy (a commonly reported range, from memory), and everything downstream inherits that ceiling.

~~~dg-fig 08.07-ceiling.svg
Held-out accuracy of a reward model trained on simulated votes. More comparisons help at first, then accuracy flattens at the ceiling: the accuracy of the true hidden reward itself on the same noisy votes.
~~~

In this simulation (where we know the true reward), noisy votes give 63 percent accuracy with 25 pairs, flattening at about 72 percent: even the true reward agrees with only 73 percent of those votes. With cleaner votes both sit near 88 percent. Near the ceiling, more data of the same kind does little; clearer guidelines or dropping ambiguous pairs raise the ceiling itself.

## Step 4: RLHF, optimise the reward but stay close

With a reward model in hand, we tune the policy $\pi_\theta$ (the language model) to produce answers it scores highly. The objective is

$$\max_{\pi}\ \mathbb E_{x,\ y \sim \pi(\cdot \mid x)}\big[r_\phi(x, y)\big] \;-\; \beta\, \mathrm{KL}\big(\pi(\cdot \mid x)\,\|\,\pi_{\text{ref}}(\cdot \mid x)\big)$$

where $\pi_{\text{ref}}$ is the **reference policy** (the supervised fine-tuned start), $\mathrm{KL}$ measures how far the answer distribution has moved from it, and $\beta > 0$ is the exchange rate between reward and distance.

Why not just maximise the reward? The reward model is only accurate near the answers it was trained on, which came from the reference. Push far away and the optimiser finds answers it scores highly for the wrong reasons: **reward hacking**, an instance of Goodhart's law (a measure that becomes a target stops being a good measure).

~~~dg-fig 08.07-hacking.svg
Left: the exact KL-regularised optimum for a range of β. The reward model's score keeps rising, but true quality peaks near 5.7 nats of KL and then falls. Right: the reason. The reward model learned "longer is better" from medium-length answers, which is wrong for very long ones.
~~~

In this small simulation each answer has a content quality and a length; true quality rises with length up to a point, then rambling hurts. The reference writes medium-length answers, so the votes only show "a bit longer is a bit better", and a reward model linear in length learns a slope of 0.75. With small $\beta$ the optimum drives answers to extreme length: the proxy score climbs to 6.6 while true quality falls from its peak of 3.3 to 1.2. Gao et al. (2023) measured the same rise-then-fall shape for real reward models.

In practice the KL is a per-token penalty $-\beta \log\big(\pi_\theta(\text{token})/\pi_{\text{ref}}(\text{token})\big)$, with the reward model's score added at the last token. PPO (lesson 08.06) optimises this with a value head and GAE over tokens: four networks in memory, namely policy, reference, reward model and critic.

## Step 5: the optimum has a closed form

For a fixed prompt, the step 4 objective has an exact solution (derivation in "Going deeper"):

$$\pi^*(y \mid x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y \mid x)\,\exp\!\big(r(x, y)/\beta\big)$$

In words: multiply each reference probability by $e^{r/\beta}$ and renormalise. $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x)\, e^{r(x,y)/\beta}$ runs over every possible text, so it cannot be computed. Small $\beta$ chases the reward; large $\beta$ stays near $\pi_{\text{ref}}$.

<div class="ex">
<div class="lab">Worked example</div>

**the optimum on three answers**

A prompt has three possible answers with reference probabilities $\pi_{\text{ref}} = (0.5, 0.3, 0.2)$ and rewards $r = (0, 1, 2)$. Take $\beta = 1$.

Unnormalised weights $\pi_{\text{ref}} e^{r}$: $0.5 \times 1 = 0.5$, $0.3 \times 2.718 = 0.816$, $0.2 \times 7.389 = 1.478$. Sum $Z = 2.793$.

Optimal policy: $\pi^* = (0.5, 0.816, 1.478)/2.793 = (0.179, 0.292, 0.529)$. Expected reward rises from $0.70$ under the reference to $1.35$, at a KL cost of 0.32 nats.

Now invert: $\beta \log(\pi^*/\pi_{\text{ref}}) = (\ln 0.358, \ln 0.973, \ln 2.645) = (-1.027, -0.027, 0.973)$. That is $r - \log Z$ with $\log Z = 1.027$: the reward, shifted by a constant.

With $\beta = 0.5$: $\pi^* = (0.037, 0.163, 0.801)$, expected reward 1.76, KL 0.92 nats. Smaller $\beta$ buys more reward with more distance.

</div>

The inversion in the last steps is the key to DPO: the log-ratio of the optimal policy to the reference *is* the reward, up to a constant that depends only on the prompt.

## Step 6: DPO, the same objective as classification

Rafailov et al. (2023) run the inversion on any policy, defining its **implicit reward** as

$$\hat r_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$

and plug it into the Bradley-Terry likelihood from step 2. The unknown $\beta \log Z(x)$ appears in both the chosen and rejected reward, so it cancels in the difference. What remains is a loss on the policy alone:

$$\mathcal L_{\text{DPO}}(\theta) = -\,\mathbb E\,\log \sigma\Big(\beta\big[(\log \pi_\theta(y_w \mid x) - \log \pi_{\text{ref}}(y_w \mid x)) - (\log \pi_\theta(y_l \mid x) - \log \pi_{\text{ref}}(y_l \mid x))\big]\Big)$$

Each $\log \pi(y \mid x)$ is the summed token log-probability of the answer. **DPO** (direct preference optimisation) is therefore ordinary supervised training: forward passes through the policy and the frozen reference, no sampling, no critic, no reward model. The policy is its own reward model, and this simplicity made DPO a default for many open models.

<div class="ex">
<div class="lab">Worked example</div>

**one DPO loss, by hand**

$\beta = 0.1$. The policy gives the chosen answer log-prob $-12$ (reference: $-13$) and the rejected answer $-15$ (reference: $-14$).

Log-ratios: chosen $-12 - (-13) = +1$, rejected $-15 - (-14) = -1$. The policy has moved toward the chosen answer and away from the rejected one.

Margin: $0.1 \times (1 - (-1)) = 0.2$.

Loss: $-\ln \sigma(0.2) = -\ln 0.550 = 0.598$ nats.

Per-pair gradient weight: $\sigma(-0.2) = 0.450$. It would be larger if the implicit reward ranked the pair the wrong way round, exactly like the reward-model loss in step 2.

</div>

## Preference learning in code

A reward model on synthetic data with a known hidden reward, then the DPO loss. The data are features, not text, but the loss lines are what a real LLM pipeline uses.

```python
import torch, torch.nn.functional as F
torch.manual_seed(0)

d, n_pairs = 8, 4000
w_true = torch.randn(d)                                  # hidden "human" reward direction
xa, xb = torch.randn(n_pairs, d), torch.randn(n_pairs, d)
a_wins = torch.rand(n_pairs) < torch.sigmoid(xa @ w_true - xb @ w_true)   # noisy BT votes
x_w = torch.where(a_wins[:, None], xa, xb)               # canonicalise: winner first
x_l = torch.where(a_wins[:, None], xb, xa)

rm = torch.nn.Sequential(torch.nn.Linear(d, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
opt = torch.optim.Adam(rm.parameters(), lr=1e-2)
tr, va = slice(0, 3000), slice(3000, None)
for step in range(300):
    margin = (rm(x_w[tr]) - rm(x_l[tr])).squeeze(-1)
    loss = -F.logsigmoid(margin).mean()                  # Bradley-Terry reward-model loss
    opt.zero_grad(); loss.backward(); opt.step()

with torch.no_grad():
    acc = (rm(x_w[va]) > rm(x_l[va])).float().mean()
    ceiling = ((x_w[va] @ w_true) > (x_l[va] @ w_true)).float().mean()
print(f"held-out accuracy {acc:.3f}, true reward {ceiling:.3f}")
# held-out accuracy 0.880, true reward 0.890

def dpo_loss(lp_w, lp_l, ref_w, ref_l, beta=0.1):       # summed token log-probs per answer
    margin = beta * ((lp_w - ref_w) - (lp_l - ref_l))
    return -F.logsigmoid(margin).mean()

print(dpo_loss(torch.tensor([-12.]), torch.tensor([-15.]),
               torch.tensor([-13.]), torch.tensor([-14.])))   # tensor(0.5981)
```

Use `F.logsigmoid`, not `torch.log(torch.sigmoid(...))`, which gives minus infinity when the sigmoid underflows. The held-out 0.880 sits just below the 0.890 ceiling from step 3.

## Reading the result: what to monitor

Preference training can look fine on its loss curve and still go wrong. Track these:

- **Reward-model accuracy on held-out pairs**, compared with annotator agreement.
- **KL from the reference**. A KL that keeps climbing is the early sign of reward hacking.
- **Chosen and rejected log-probabilities separately** during DPO, not only their difference.
- **Answer length**. A steady upward drift is the most common symptom of a hacked reward.
- **A held-out judgement** the reward model never saw: fresh human comparisons, or win rate judged by a separate strong model.

The third point deserves a figure. In the toy DPO run below (a softmax policy over 400 candidate answers described by 12 features), the pair accuracy of the implicit reward reaches 83 percent and true quality rises from 0.00 to 2.71. Yet the average log-probability of the chosen answers falls from $-5.5$ to $-30$.

~~~dg-fig 08.07-dpo.svg
Left: during DPO with β = 0.1 both chosen and rejected log-probabilities fall; the loss only needs the rejected ones to fall faster. Right: KL from the reference over training. Smaller β ends further from the reference and with higher true quality in this well-specified toy.
~~~

The loss depends only on the *margin*, so it is satisfied by pushing both answers down as long as the rejected one drops faster. Mass moves to answers the dataset never mentions: good ones in this toy, where features capture quality exactly, but possibly off-distribution or degenerate text in a real model. The right panel shows $\beta$ as the leash: final KL is 5.9, 2.9, 1.2 and 0.4 nats for $\beta$ = 0.1, 0.5, 1 and 2.

## What goes wrong

| Problem | Symptom | Fix
| --- | --- | --- |
| Reward hacking | proxy reward and KL keep rising, human ratings fall | larger $\beta$, early stopping, refresh the reward model on new policy samples, ensembles of reward models
| Length bias | answers get longer every round; reward correlates with length | length-controlled evaluation, length penalty, pairs matched in length
| Label noise | reward-model accuracy stuck near 65 to 75 percent | clearer guidelines, multiple annotators per pair, drop ambiguous pairs
| Inconsistent pair order | loss behaves oddly; accuracy near 50 percent | canonicalise every pair so the winner is first; drop or model ties separately
| DPO likelihood collapse | chosen log-probs fall, outputs drift or degrade | higher $\beta$, fewer epochs, add a supervised (NLL) term on chosen answers
| Off-policy staleness | DPO gains stall after the policy changes | regenerate pairs from the current policy (iterative or online DPO)

A typical DPO setting is $\beta \approx 0.1$. For tasks with checkable answers, such as maths and code, a verifier can replace the learned reward model altogether; that setting (RLVR, usually with GRPO) is covered in the AI Research course.

## Trade-offs: RLHF versus DPO

| Aspect | RLHF with PPO | DPO
| --- | --- | --- |
| Data | comparisons for the reward model, then fresh on-policy samples | one fixed set of comparisons
| Networks in memory | policy, reference, reward model, critic | policy, reference
| Strength | on-policy exploration; reusable reward model | simple, stable, cheap
| Typical failure | reward hacking, PPO instability | overfits the pairs; can lower the likelihood of both answers; off-policy drift

## Going deeper

**Deriving the closed form.** For one prompt, write the objective as $\sum_y \pi(y)\big[r(y) - \beta \log(\pi(y)/\pi_{\text{ref}}(y))\big]$. Divide by $\beta$ and rearrange: it equals $-\mathrm{KL}(\pi \,\|\, \pi^*) + \log Z$ with $\pi^*$ as in step 5. Since KL is never negative and is zero only when the two distributions match, the maximum is $\pi = \pi^*$, and the maximum value is $\beta \log Z$. This is the same Gibbs-distribution argument as in maximum-entropy RL, with the reference policy playing the role of the prior.

**A Bayesian reading.** The optimum is a Bayesian update: the reference is the prior and $e^{r/\beta}$ the likelihood. If the reward model's error grows with distance from its data, there is a best $\beta$ (the peak in the hacking figure), found only with a held-out judge.

**What DPO does not guarantee.** DPO and RLHF share an optimum only with data covering all answers. With finite data DPO constrains only answers in the pairs, while RLHF at least scores what the policy actually writes. Hence on-policy and iterative DPO variants.

## Key terms

| Term | Meaning
| --- | --- |
| Preference pair | A prompt with a chosen answer $y_w$ and a rejected answer $y_l$
| Bradley-Terry model | $P(y_w \succ y_l) = \sigma(r_w - r_l)$: win probability is a sigmoid of the reward gap
| Reward model | A network trained on preference pairs to output a scalar score for any prompt and answer
| Reference policy | The starting (supervised fine-tuned) model that the KL penalty keeps the policy near
| KL penalty, $\beta$ | Cost of moving away from the reference; larger $\beta$ means a shorter leash
| RLHF | Fit a reward model to human comparisons, then optimise the policy against it with RL (usually PPO)
| Reward hacking | The policy exploits errors in the learned reward, so proxy reward rises while true quality falls
| DPO | Direct preference optimisation: a logistic loss on policy-to-reference log-ratios of chosen versus rejected answers
| Implicit reward | $\beta \log(\pi_\theta / \pi_{\text{ref}})$, the reward a DPO-trained policy implicitly represents

<div class="take">
<div class="lab">Key takeaway</div>

preference learning fits a Bradley-Terry reward model to pairwise human comparisons and optimises a policy against it under a KL leash to the reference model; DPO solves the same KL-regularised objective in closed form, turning it into a logistic loss on log-probability ratios of preferred versus rejected answers, and in both cases the key control against reward hacking is $\beta$.

</div>
