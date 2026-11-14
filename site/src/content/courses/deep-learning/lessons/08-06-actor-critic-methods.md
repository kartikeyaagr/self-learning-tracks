---
id: "08.06"
course: "deep-learning"
module: "08"
slug: "actor-critic-methods"
title: "Actor-critic methods"
keyIdeas: ["Policy-gradient methods optimise a parameterised policy directly: $\\nabla J=\\mathbb E[\\nabla_\\theta\\log\\pi_\\theta(a\\mid s)\\,A(s,a)]$, with the advantage $A$ telling each action whether it beat expectations.", "An actor-critic pairs the policy (actor) with a learned value function (critic) that supplies a low-variance advantage estimate, such as the TD error $r+\\gamma V(s')-V(s)$.", "Generalised Advantage Estimation is the $\\lambda$-weighted sum of TD errors, $\\hat A_t=\\sum_l(\\gamma\\lambda)^l\\delta_{t+l}$, computed backwards in one pass; $\\lambda$ trades bias for variance.", "The A2C loss combines a policy term, a value regression term and an entropy bonus that keeps the policy from collapsing to deterministic too early.", "PPO adds a clipped importance ratio to reuse each batch for several epochs, and GRPO drops the critic for group-relative advantages; both are covered in AI Research."]
question: "Your actor-critic implementation computes `loss = -(logp * (returns - values)).mean() + 0.5 * ((returns - values)**2).mean()` with a single shared network, and training is unstable: the policy seems to chase the critic. What is the bug, and what else would you check in this loss?"
answer: "The advantage `returns - values` is not detached, so the policy term backpropagates into the critic. Gradient descent on $-\\log\\pi\\cdot(R-V)$ with respect to $V$ pushes $V$ up whenever $\\log\\pi$ is negative (always, for a stochastic policy), and in general moves the critic to make the policy loss small rather than to predict returns. With a shared body, the policy and value objectives then fight through the same features. Fix: `adv = (returns - values).detach()`, optionally normalised per batch, and use the non-detached values only in the value loss. Other checks: add an entropy bonus (missing here) so the policy does not collapse; make sure `returns` are computed with bootstrapping from $V(s_{T})$ at truncation and none at true terminals; consider GAE instead of raw returns for lower variance; and check the relative scale of the value loss, since on a shared network a large value loss can swamp the policy gradient."
sources: [{"title": "Schulman et al.: High-Dimensional Continuous Control Using Generalized Advantage Estimation", "url": "https://arxiv.org/abs/1506.02438", "note": "GAE: the derivation as a sum of TD errors and the bias-variance role of lambda."}, {"title": "Mnih et al.: Asynchronous Methods for Deep Reinforcement Learning", "url": "https://arxiv.org/abs/1602.01783", "note": "A3C: the actor-critic loss with an entropy bonus, the basis of A2C."}, {"title": "OpenAI Spinning Up: Intro to Policy Optimization", "url": "https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html", "note": "The policy gradient theorem, baselines and advantage functions, with code."}, {"title": "Schulman et al.: Proximal Policy Optimization Algorithms", "url": "https://arxiv.org/abs/1707.06347", "note": "Where actor-critic goes next: the clipped surrogate used with GAE."}]
practice: ["actor-critic-a2c-loss-with-entropy", "gae-advantages", "reinforce-with-baseline", "reinforce-gradient"]
tags: ["actor-critic", "GAE", "policy gradient"]
tools: ["NumPy", "PyTorch"]
---

## Why learn the policy directly

Q-learning (08.04) learns values and acts greedily. That struggles with continuous actions (the max over actions becomes an optimisation problem), cannot represent stochastic policies that are genuinely optimal (in games with hidden information, or when aliasing makes two states look the same), and small value changes can flip the greedy action discontinuously. **Policy-gradient methods** parameterise $\pi_\theta(a\mid s)$, for example a softmax over action logits or a Gaussian over continuous actions, and do gradient ascent on expected return $J(\theta)$.

The **policy gradient theorem** gives the gradient without differentiating the environment:

$$\nabla_\theta J=\mathbb E_{\pi_\theta}\Big[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\Psi_t\Big]$$

where $\Psi_t$ can be the return $G_t$ (REINFORCE), $G_t-b(s_t)$ for any baseline $b$, or the advantage $A^\pi(s_t,a_t)$. All give the same expected gradient, because $\mathbb E_a[\nabla\log\pi(a\mid s)]\,b(s)=0$, but very different variance. (The AI Research course derives REINFORCE; here we focus on the critic.)

## The critic and the advantage

The best simple baseline is $V^\pi(s)$, which turns $\Psi_t$ into an advantage: did this action do better than the policy's average from this state? An **actor-critic** learns $V_\phi(s)$ alongside the policy, using TD methods from lesson 08.03, and estimates the advantage with the TD error:

$$\hat A_t=\delta_t=r_{t+1}+\gamma V_\phi(s_{t+1})-V_\phi(s_t)$$

This is low-variance (one reward, one transition) but biased whenever $V_\phi$ is wrong. Using the full return minus $V_\phi(s_t)$ is unbiased but noisy. That is the same bias-variance dial as TD versus Monte Carlo, and GAE turns it continuously.

~~~dg-flow The critic turns raw rewards into advantages; the actor follows advantage-weighted log-probability gradients.
Rollout s, a, r -> *Critic V -(TD errors, GAE)-> Advantages -(policy gradient)-> Actor update
Rollout s, a, r -(value targets)-> *Critic V
~~~

## Generalised Advantage Estimation

Schulman et al. (2015) define the advantage as an exponentially weighted average of n-step advantages, which collapses to a weighted sum of TD errors:

$$\hat A_t^{\text{GAE}(\gamma,\lambda)}=\sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l}$$

$\lambda=0$ gives the one-step TD advantage; $\lambda=1$ gives $G_t-V(s_t)$. Typical settings are $\gamma=0.99$, $\lambda=0.95$. It is computed backwards over a rollout in a single pass, resetting at episode boundaries:

```python
def gae(rewards, values, last_value, dones, gamma=0.99, lam=0.95):
    adv = np.zeros_like(rewards); next_adv = 0.0; next_v = last_value
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        next_adv = delta + gamma * lam * nonterminal * next_adv
        adv[t] = next_adv; next_v = values[t]
    return adv, adv + values              # advantages and value targets
```

<div class="ex">
<div class="lab">Worked example</div>

**GAE over three steps**

$\gamma=0.9$, $\lambda=0.8$ (so $\gamma\lambda=0.72$); rewards $(1, 0, 1)$; values $V=(0.5, 0.4, 0.6)$; the episode ends after step 3.

TD errors: $\delta_2=1-0.6=0.4$; $\delta_1=0+0.9\cdot0.6-0.4=0.14$; $\delta_0=1+0.9\cdot0.4-0.5=0.86$.

Advantages backwards: $\hat A_2=0.4$; $\hat A_1=0.14+0.72\cdot0.4=0.428$; $\hat A_0=0.86+0.72\cdot0.428=1.168$.

Value targets $\hat A+V=(1.668, 0.828, 1.0)$; the last is exactly the final reward, as it should be at a terminal.

</div>

## The A2C loss

Advantage actor-critic (A2C, the synchronous form of A3C) minimises one combined loss per batch:

$$\mathcal L=-\frac1N\sum_t\log\pi_\theta(a_t\mid s_t)\,\hat A_t\;+\;c_v\frac1N\sum_t\big(V_\phi(s_t)-\hat R_t\big)^2\;-\;c_e\frac1N\sum_t\mathcal H\big(\pi_\theta(\cdot\mid s_t)\big)$$

- **Policy term.** $\hat A_t$ is treated as a constant (no gradient through the critic into the actor's term). Normalising advantages per batch to zero mean and unit variance is common and stabilises step sizes.
- **Value term.** Regress $V_\phi$ onto the targets $\hat R_t=\hat A_t+V(s_t)$; $c_v\approx0.5$. Actor and critic often share a network body with two heads, the same design as AlphaGo Zero's policy-value network (09.03).
- **Entropy bonus.** $\mathcal H(\pi)=-\sum_a\pi(a\mid s)\log\pi(a\mid s)$ rewards keeping probability spread across actions; $c_e\approx0.01$. Without it, the policy often becomes deterministic early on a mediocre action and stops exploring. Too large and it never commits.

## Failure modes and where this leads

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Entropy collapses in the first few hundred updates | step size too large, no entropy bonus | lower LR, raise $c_e$, normalise advantages
| Value loss dominates, policy barely moves | rewards on a large scale | normalise rewards or returns, tune $c_v$
| Learning stalls on long-horizon tasks | $\lambda$ or $\gamma$ too small, critic biased | raise $\lambda$, check bootstrapping at truncation
| High variance, unstable | $\lambda$ near 1, small batches | lower $\lambda$, more parallel environments

A2C uses each batch for one gradient step, because the policy gradient is only valid for data from the current policy. **PPO** reuses a batch for several epochs by weighting with the importance ratio $\pi_\theta/\pi_{\text{old}}$ and clipping it to $[1-\epsilon,1+\epsilon]$; **GRPO**, used for LLM reasoning, removes the critic and computes advantages relative to a group of samples for the same prompt. Both are covered in the AI Research course's Reinforcement Learning chapter, and RLHF with PPO is the subject of the next lesson.

<div class="take">
<div class="lab">Key takeaway</div>

an actor-critic follows the policy gradient weighted by advantages from a learned critic; GAE mixes multi-step TD errors with weight $(\gamma\lambda)^l$ to set the bias-variance balance, and the A2C loss adds value regression and an entropy bonus that keeps exploration alive.

</div>
