---
id: "08.07"
course: "deep-learning"
module: "08"
slug: "learning-from-preferences"
title: "Learning from preferences"
keyIdeas: ["When a reward is hard to write down, collect pairwise human judgements instead and fit a reward model; comparisons are easier and more consistent for people than absolute scores.", "The Bradley-Terry model sets $P(y_w\\succ y_l)=\\sigma(r(y_w)-r(y_l))$; the reward model is trained with $-\\log\\sigma(r_w-r_l)$, and only reward differences are identified.", "RLHF maximises $\\mathbb E[r_\\phi(x,y)]-\\beta\\,\\mathrm{KL}(\\pi\\|\\pi_\\text{ref})$ with PPO; the KL keeps the policy where the reward model is trustworthy.", "DPO uses the closed-form optimum of that objective to turn it into a classification loss on preference pairs, $-\\log\\sigma\\big(\\beta[\\Delta_w-\\Delta_l]\\big)$ with $\\Delta=\\log\\pi-\\log\\pi_\\text{ref}$; no reward model or sampling needed.", "Reward hacking (over-optimising a proxy), length bias and label noise are the main failure modes; the KL coefficient $\\beta$ is the key control."]
question: "After DPO training, your model's win rate against the reference improves, but you notice the log-probability of the chosen answers on the training set went down, not up. Is this a bug? Explain using the DPO loss, and say what risk it signals."
answer: "It is not necessarily a bug. The DPO loss depends only on the margin $\\beta[(\\log\\pi(y_w)-\\log\\pi_\\text{ref}(y_w))-(\\log\\pi(y_l)-\\log\\pi_\\text{ref}(y_l))]$. It can be increased by pushing the rejected answer's likelihood down faster than the chosen one's, even while both fall. The chosen and rejected answers for a prompt are usually similar, sharing most tokens, so gradients that push down $y_l$ also push down $y_w$; the loss is satisfied as long as $y_l$ falls more. The risk is that probability mass leaves both answers and goes somewhere the data says nothing about, possibly to off-distribution or degenerate outputs, which is a form of over-optimisation that the implicit KL does not fully prevent with a finite dataset. Monitor chosen and rejected log-probabilities and KL from the reference; mitigations include a higher $\\beta$, early stopping, adding an SFT (negative log-likelihood) term on chosen answers, or on-policy variants that refresh the pairs."
sources: [{"title": "Christiano et al.: Deep Reinforcement Learning from Human Preferences", "url": "https://arxiv.org/abs/1706.03741", "note": "The original reward-model-from-comparisons method on Atari and MuJoCo."}, {"title": "Ouyang et al.: Training Language Models to Follow Instructions with Human Feedback", "url": "https://arxiv.org/abs/2203.02155", "note": "InstructGPT: SFT, reward modelling and PPO with a per-token KL penalty."}, {"title": "Rafailov et al.: Direct Preference Optimization", "url": "https://arxiv.org/abs/2305.18290", "note": "The closed-form optimum and the derivation of the DPO loss (sections 4 and 5)."}, {"title": "Nathan Lambert: Reinforcement Learning from Human Feedback (book)", "url": "https://rlhfbook.com/", "note": "Free book covering reward models, regularisation, PPO and direct alignment methods in practice."}]
practice: ["bradley-terry-reward-model-loss", "dpo-closed-form-loss", "canonicalize-pairwise-preference-data", "ppo-with-reference-policy-kl", "score-kl-regularized-policy-batch"]
tags: ["RLHF", "Bradley-Terry", "DPO"]
tools: ["PyTorch"]
---

## When you cannot write the reward

Every method so far assumed a reward function. For many tasks nobody can write one: "a helpful, harmless answer", "a backflip that looks natural", "a good summary". People can, however, look at two outputs and say which is better, faster and more consistently than they can assign absolute scores. **Learning from preferences** turns those comparisons into a reward model, then optimises a policy against it. Christiano et al. (2017) did this for Atari and simulated robots with under 1 percent of interactions labelled; InstructGPT (2022) applied it to language models, which made RLHF the standard final stage of LLM training. (The AI Research add-on on training LLMs at scale covers the full post-training pipeline; this lesson covers the objectives.)

~~~dg-flow The RLHF pipeline: people compare, a reward model generalises their judgements, RL optimises against it.
Prompts -(sample two answers)-> Human comparisons -(Bradley-Terry fit)-> *Reward model -(PPO with KL)-> Tuned policy
~~~

## Bradley-Terry: from comparisons to a reward

The **Bradley-Terry model** says each item has a latent score and the probability that one beats another is a logistic function of the difference:

$$P(y_w\succ y_l\mid x)=\sigma\big(r(x,y_w)-r(x,y_l)\big)$$

Fit a reward model $r_\phi$ (for LLMs, the pretrained model with its unembedding replaced by a scalar head, read at the last token) by maximum likelihood on comparison data:

$$\mathcal L_{\text{RM}}=-\,\mathbb E_{(x,y_w,y_l)}\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)$$

Only differences enter, so adding a constant to all rewards changes nothing: the reward is identified only up to a per-prompt shift, and its absolute values are meaningless until you normalise them. Practical details matter: canonicalise pairs so the winner is always first, drop ties or model them separately, deduplicate, and sometimes add a margin when annotators rate one answer as much better. Reward-model accuracy on held-out pairs is typically only 65 to 75 percent, because human labels are noisy; this is the ceiling the rest of the pipeline inherits.

<div class="ex">
<div class="lab">Worked example</div>

**one reward-model loss**

For a prompt, the reward model scores the preferred answer 1.2 and the rejected one 0.4.

$P(\text{preferred wins})=\sigma(0.8)=0.69$, so the loss is $-\ln0.69=0.37$ nats.

The gradient pushes the two scores apart with weight $1-0.69=0.31$; a pair already separated by 5 gets weight 0.007, so easy pairs barely contribute.

</div>

## RLHF: optimise the reward, but stay close

With a reward model, the policy (the LLM) is tuned by RL to maximise

$$\max_\pi\ \mathbb E_{x,\,y\sim\pi}\big[r_\phi(x,y)\big]-\beta\,\mathrm{KL}\big(\pi(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big)$$

where $\pi_{\text{ref}}$ is the supervised fine-tuned starting model. The KL term is essential. The reward model is only accurate near the distribution it was trained on, so an unconstrained optimiser finds outputs it scores highly for the wrong reasons: **reward hacking**, an instance of Goodhart's law. Gao et al. measured that true quality first rises and then falls as KL from the reference grows, while proxy reward keeps rising. In practice the KL is added per token as a penalty $-\beta\log(\pi/\pi_\text{ref})$ to the reward, and PPO (an actor-critic from lesson 08.06) does the optimisation, with a value head and GAE over tokens. That means four models in memory: policy, reference, reward model and critic.

## DPO: the same objective as classification

Rafailov et al. (2023) observed that the KL-regularised objective has a closed-form optimum:

$$\pi^*(y\mid x)=\frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\big(r(x,y)/\beta\big)\ \Longleftrightarrow\ r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)}+\beta\log Z(x)$$

Substituting this reward into the Bradley-Terry likelihood, the intractable $\log Z(x)$ cancels in the difference, leaving a loss on the policy alone:

$$\mathcal L_{\text{DPO}}=-\,\mathbb E\,\log\sigma\Big(\beta\big[(\log\pi_\theta(y_w\mid x)-\log\pi_{\text{ref}}(y_w\mid x))-(\log\pi_\theta(y_l\mid x)-\log\pi_{\text{ref}}(y_l\mid x))\big]\Big)$$

```python
def dpo_loss(lp_w, lp_l, ref_w, ref_l, beta=0.1):   # summed token log-probs per sequence
    margin = beta * ((lp_w - ref_w) - (lp_l - ref_l))
    return -torch.nn.functional.logsigmoid(margin).mean()
```

The policy is its own implicit reward model. Training is supervised: two forward passes per pair (policy and frozen reference), no sampling, no critic, no reward model. That simplicity made DPO and its variants the default for many open models.

<div class="ex">
<div class="lab">Worked example</div>

**one DPO loss**

$\beta=0.1$. The policy gives the chosen answer log-prob $-12$ (reference $-13$) and the rejected answer $-15$ (reference $-14$).

Implicit reward differences: chosen $+1$, rejected $-1$; margin $=0.1\cdot(1-(-1))=0.2$.

Loss $=-\ln\sigma(0.2)=0.598$; the per-pair gradient weight is $\sigma(-0.2)=0.45$, larger when the implicit reward model ranks the pair wrongly.

</div>

## Trade-offs and failure modes

| Aspect | RLHF with PPO | DPO
| --- | --- | --- |
| Data | comparisons for the RM, then fresh on-policy samples | a fixed set of comparisons
| Models in memory | policy, reference, reward model, critic | policy, reference
| Strength | on-policy exploration; reward model reusable | simple, stable, cheap
| Typical failure | reward hacking, PPO instability | overfits the pairs; can lower the likelihood of both answers; off-policy drift

Common to both: **length bias** (annotators and reward models favour longer answers, so policies get verbose), sensitivity to label noise, and the choice of $\beta$ (around 0.1 for DPO; smaller allows larger departures from the reference). For tasks with checkable answers, such as maths and code, verifiable rewards remove the learned reward model altogether; that RLVR setting, with GRPO, is covered in the AI Research course.

<div class="take">
<div class="lab">Key takeaway</div>

preference learning fits a Bradley-Terry reward model to pairwise human comparisons and optimises a policy against it under a KL leash to the reference model; DPO solves the same KL-regularised objective in closed form, turning it into a logistic loss on log-probability ratios of preferred versus rejected answers.

</div>
