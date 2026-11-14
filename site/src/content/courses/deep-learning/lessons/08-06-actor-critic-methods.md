---
id: "08.06"
course: "deep-learning"
module: "08"
slug: "actor-critic-methods"
title: "Actor-critic methods"
keyIdeas: ["Policy-gradient methods learn the policy $\\pi_\\theta(a\\mid s)$ directly and ascend $\\nabla J=\\mathbb E[\\nabla_\\theta\\log\\pi_\\theta(a\\mid s)\\,\\Psi]$: push up actions whose weight $\\Psi$ is positive, push down the rest. This handles continuous actions and stochastic policies that greedy Q-learning cannot.", "Subtracting a state-only baseline never changes the expected gradient but can slash its variance; the best simple baseline is $V(s)$, which turns the weight into an **advantage**: did this action beat the policy's average?", "An actor-critic learns the critic $V_\\phi$ with TD and uses the TD error $r+\\gamma V(s')-V(s)$ as a low-variance advantage, at the cost of bias whenever the critic is wrong.", "GAE, $\\hat A_t=\\sum_l(\\gamma\\lambda)^l\\delta_{t+l}$, is computed backwards in one pass; $\\lambda=0$ is one-step TD (low variance, biased), $\\lambda=1$ is return minus value (unbiased, noisy), and $\\lambda\\approx0.95$ is a common middle.", "The A2C loss is policy term + $c_v\\cdot$value regression $-$ $c_e\\cdot$entropy; the advantage must be detached, and the entropy bonus should be small enough to let the policy commit but large enough to stop early collapse.", "Actor-critic is not automatically better than a good Monte Carlo baseline (in our CartPole run it was not); check several seeds, entropy, value loss and critic explained variance. PPO and GRPO extend these ideas and are covered in AI Research."]
question: "Your actor-critic implementation computes `loss = -(logp * (returns - values)).mean() + 0.5 * ((returns - values)**2).mean()` with a single shared network, and training is unstable: the policy seems to chase the critic. What is the bug, and what else would you check in this loss?"
answer: "The advantage `returns - values` is not detached, so the policy term backpropagates into the critic. Gradient descent on $-\\log\\pi\\cdot(R-V)$ with respect to $V$ pushes $V$ up whenever $\\log\\pi$ is negative (always, for a stochastic policy), and in general moves the critic to make the policy loss small rather than to predict returns. With a shared body, the policy and value objectives then fight through the same features. Fix: `adv = (returns - values).detach()`, optionally normalised per batch to zero mean and unit variance, and use the non-detached values only in the value loss. Other checks: add an entropy bonus (missing here, typically $c_e\\approx0.01$) so the policy does not collapse to a deterministic action early; make sure `returns` are computed with bootstrapping from $V(s_T)$ at time-limit truncation and with zero only at true terminals; consider GAE ($\\lambda\\approx0.95$) instead of raw returns for lower variance; check the relative scale of the value loss, since on a shared network a large value loss can swamp the policy gradient (lower $c_v$, normalise returns, or split the networks); and make sure each batch is used for only one update, since the plain policy gradient is on-policy. Finally, monitor entropy, value loss and the critic's explained variance over several seeds rather than one learning curve."
sources: [{"title": "Schulman et al.: High-Dimensional Continuous Control Using Generalized Advantage Estimation", "url": "https://arxiv.org/abs/1506.02438", "note": "GAE: the derivation as a sum of TD errors and the bias-variance role of lambda."}, {"title": "Mnih et al.: Asynchronous Methods for Deep Reinforcement Learning", "url": "https://arxiv.org/abs/1602.01783", "note": "A3C: the actor-critic loss with an entropy bonus, the basis of A2C."}, {"title": "OpenAI Spinning Up: Intro to Policy Optimization", "url": "https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html", "note": "The policy gradient theorem, baselines and advantage functions, with code."}, {"title": "Schulman et al.: Proximal Policy Optimization Algorithms", "url": "https://arxiv.org/abs/1707.06347", "note": "Where actor-critic goes next: the clipped surrogate used with GAE."}]
practice: ["actor-critic-a2c-loss-with-entropy", "gae-advantages", "reinforce-with-baseline", "reinforce-gradient"]
tags: ["actor-critic", "GAE", "policy gradient"]
tools: ["NumPy", "PyTorch"]
---

## What you will learn

- Why you might learn a policy directly instead of learning Q-values and acting greedily.
- The policy gradient theorem in plain words: push up the log-probability of actions that did better than expected.
- What the critic adds: a learned value function that turns noisy returns into low-variance **advantages**, and the price you pay for it (bias).
- How Generalised Advantage Estimation (GAE) mixes one-step and many-step estimates with one dial $\lambda$.
- The three-part A2C loss (policy, value, entropy), how to code it in PyTorch, and the bugs that break it most often.

*Before you start:* returns and value functions (08.01), TD errors (08.03), Q-learning (08.04), and softmax with log-probabilities.

Lesson 08.04 learned action values and acted greedily; 08.05 asked how to keep exploring. Here we learn the policy itself, keeping a value function only as a helper that grades the policy's choices.

## The idea in one picture

A robot stands at a junction with three choices: left, straight, right. Its **policy** (the **actor**) is a softmax over three numbers called logits; all three start at 0, so each action has probability 1/3. A second component, the **critic**, has learned that being at the junction is worth $V = 4$ on average. The discount is $\gamma = 0.9$.

The robot tries two actions and records what happened:

| Action tried | Reward $r$ | Critic's value of the next state | Target $r + 0.9\,V(s')$ | TD error $\delta$ = target $-$ 4 | Verdict
| --- | --- | --- | --- | --- | --- |
| right | 1 | 6 | 6.4 | +2.4 | better than expected
| left | 0 | 2 | 1.8 | $-2.2$ | worse than expected

The TD error $\delta$ is the critic's verdict: "you ended up better (or worse) off than I predicted". The actor nudges its logits toward the tried action, scaled by $\delta$: a positive verdict raises it, a negative one lowers it.

~~~dg-fig 08.06-junction.svg
The actor's probabilities at the junction. A +2.4 verdict for "right" lifts it from 0.33 to 0.62; a −2.2 verdict for "left" then cuts left to 0.05. "Straight" was never tried, yet it also drops, because probabilities must sum to one.
~~~

The actor never needed the reward to be large in absolute terms, only whether the action beat the critic's expectation; that comparison is the **advantage**. And the verdict came after one step, because the critic's value of the next state stood in for everything afterwards. An actor that chooses plus a critic that grades is the whole idea of **actor-critic methods**.

## Step 1: why learn the policy directly

Q-learning learns $Q(s,a)$ and acts with $\arg\max_a Q(s,a)$. That works well for small discrete action sets, but it has three weak spots:

- **Continuous actions.** For a steering angle, "max over actions" is an optimisation problem at every step.
- **Stochastic policies.** With hidden information (rock-paper-scissors) or states that look identical, the best policy can be random; greedy cannot represent it.
- **Jumpy behaviour.** A tiny change in Q-values can flip the argmax and change behaviour completely.

**Policy-gradient methods** avoid all three. They give the policy its own parameters $\theta$ and write it as $\pi_\theta(a \mid s)$, the probability of action $a$ in state $s$. Discrete actions use a softmax over logits; continuous ones usually a Gaussian whose mean the network outputs. Then they do gradient *ascent* on the expected discounted return

$$J(\theta) = \mathbb E_{\pi_\theta}\Big[\sum_t \gamma^t r_{t+1}\Big].$$

## Step 2: the policy gradient theorem

$J$ depends on the environment, which we cannot differentiate. The **policy gradient theorem** shows we never need to:

$$\nabla_\theta J = \mathbb E_{\pi_\theta}\Big[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\,\Psi_t\Big].$$

Symbol by symbol:

- $\log \pi_\theta(a_t \mid s_t)$ is the log-probability of the action actually taken; its gradient points where that action becomes more likely.
- $\Psi_t$ is a scalar weight: positive means "do this more", negative "do this less".
- The expectation is over the current policy's trajectories, approximated by the ones just collected.

For a softmax policy the gradient with respect to the logits has a tidy form: $\nabla \log \pi(a) = e_a - \pi$, the one-hot vector of the chosen action minus the probability vector. In the junction, the first update used $e_{\text{right}} - \pi = (-\tfrac13, -\tfrac13, \tfrac23)$, multiplied by step size $0.5$ and $\delta = 2.4$, giving new logits $(-0.4, -0.4, 0.8)$ and probabilities $(0.19, 0.19, 0.62)$.

Several choices of $\Psi_t$ give the same expected gradient:

1. The return $G_t$ from step $t$ onward. This is **REINFORCE**.
2. The return minus a **baseline** that depends only on the state, $G_t - b(s_t)$.
3. The **advantage** $A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)$, or any unbiased estimate of it.

Why does subtracting a baseline not change the expectation? Because $\mathbb E_{a \sim \pi}[\nabla \log \pi(a \mid s)] = \sum_a \nabla \pi(a \mid s) = \nabla \sum_a \pi(a \mid s) = \nabla 1 = 0$, so $\mathbb E[\nabla \log \pi \cdot b(s)] = b(s) \cdot 0 = 0$. The baseline removes nothing from the average push, but it can remove a great deal of noise.

## Step 3: what the critic buys you (variance)

The return $G_t$ adds up every future reward, each with its own randomness, so REINFORCE gradients are noisy.

Take a 10-cell corridor: start in cell 1, goal in cell 9, each step costs $-1$ plus Gaussian noise (standard deviation 2), $\gamma = 0.95$. Fix a policy that goes right with probability 0.6 and compute, over 8000 episodes, each episode's estimate of one gradient component (the "go right" logit in cell 1). The exact $V^\pi$ is known here, so every estimator is unbiased.

~~~dg-fig 08.06-variance.svg
Same policy, same quantity, four ways of weighting the log-probability gradient. All four histograms centre near 0.5, but the spread falls from 17.1 (raw returns) to 7.2 (minus a baseline), 4.3 (GAE) and 2.2 (one-step TD error).
~~~

The averages agree up to sampling noise, but the spreads do not: the raw-return estimator needs roughly $(17.1/2.2)^2 \approx 60$ times as many episodes as the TD-error one for the same precision.

An **actor-critic** learns the critic $V_\phi(s)$, a value network with its own parameters $\phi$, alongside the policy, and estimates the advantage with the TD error from 08.03:

$$\hat A_t = \delta_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t).$$

Here $r_{t+1}$ is the reward right after the action, and $V_\phi(s_{t+1})$ stands in for all future rewards. It has low variance (one random reward, one transition) but is unbiased only if $V_\phi$ is exactly right, which a learned critic never is. The return minus $V_\phi(s_t)$ stays unbiased as a gradient even with a bad critic, but is noisy: the Monte Carlo versus TD trade-off of 08.03 again.

~~~dg-flow The critic turns raw rewards into advantages; the actor follows advantage-weighted log-probability gradients.
Rollout s, a, r -> *Critic V -(TD errors, GAE)-> Advantages -(policy gradient)-> Actor update
Rollout s, a, r -(value targets)-> *Critic V
~~~

## Step 4: GAE, one dial between TD and Monte Carlo

Schulman et al. (2015) average the n-step advantage estimates (n real rewards, then bootstrap) with exponentially decaying weights. The result collapses to a weighted sum of TD errors:

$$\hat A_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l\, \delta_{t+l}.$$

Here $\delta_{t+l}$ is the TD error $l$ steps later and $\lambda \in [0, 1]$ is the dial:

- $\lambda = 0$ keeps only $\delta_t$: the one-step TD advantage, low variance, biased by the critic.
- $\lambda = 1$ keeps every TD error with weight $\gamma^l$; the critic's values cancel in a telescoping sum and you get $G_t - V(s_t)$: unbiased as a gradient, high variance.
Typical settings are $\gamma = 0.99$ and $\lambda = 0.95$.

~~~dg-fig 08.06-gae.svg
Left: how much GAE weights a TD error l steps ahead. Right: on the corridor with a deliberately poor critic (values 30% of the true ones), squared bias of the gradient falls from 25.6 to 0.2 as λ goes from 0 to 1 while variance rises from 2.6 to 71; the total is lowest near λ = 0.75.
~~~

The right panel uses the same fixed policy, but the critic reports only 30% of each true value. At $\lambda = 0$ the estimate inherits that error; at $\lambda = 1$ the error cancels but the return's noise comes back. A better critic moves the best $\lambda$ down; a worse one moves it up.

GAE is computed backwards over a rollout in a single pass, because $\hat A_t = \delta_t + \gamma\lambda\,\hat A_{t+1}$. At a true terminal state the next value is 0 and the recursion resets:

(The full code below contains this backward loop as the function `gae`.)

Advantage plus value is the **value target** $\hat R_t$ the critic is trained to predict.

<div class="ex">
<div class="lab">Worked example</div>

**GAE over three steps, by hand**

Settings: $\gamma = 0.9$, $\lambda = 0.8$, so $\gamma\lambda = 0.72$. Rewards $(1, 0, 1)$; the critic's values $V = (0.5, 0.4, 0.6)$; the episode ends after the third step, so the value after it is 0.

TD errors, working backwards. Step 2: $\delta_2 = 1 + 0.9 \cdot 0 - 0.6 = 0.4$. Step 1: $\delta_1 = 0 + 0.9 \cdot 0.6 - 0.4 = 0.54 - 0.4 = 0.14$. Step 0: $\delta_0 = 1 + 0.9 \cdot 0.4 - 0.5 = 1.36 - 0.5 = 0.86$.

Advantages backwards with $\hat A_t = \delta_t + 0.72\,\hat A_{t+1}$: $\hat A_2 = 0.4$; $\hat A_1 = 0.14 + 0.72 \cdot 0.4 = 0.14 + 0.288 = 0.428$; $\hat A_0 = 0.86 + 0.72 \cdot 0.428 = 0.86 + 0.308 = 1.168$.

Value targets $\hat A + V$: $(1.168 + 0.5,\ 0.428 + 0.4,\ 0.4 + 0.6) = (1.668,\ 0.828,\ 1.0)$. The last one is exactly the final reward, as it must be at a terminal state.

Check the extremes. With $\lambda = 0$ the advantages are just the TD errors $(0.86, 0.14, 0.4)$. With $\lambda = 1$ the returns are $G = (1 + 0 + 0.81,\ 0.9,\ 1) = (1.81, 0.9, 1)$ and $G - V = (1.31, 0.5, 0.4)$, which the recursion with $\gamma\lambda = 0.9$ reproduces exactly. Our $\lambda = 0.8$ answer, 1.168 for the first step, sits between 0.86 and 1.31.

</div>

## Step 5: the A2C loss

**Advantage actor-critic (A2C)**, the synchronous form of A3C (Mnih et al., 2016), minimises one combined loss per batch of $N$ time steps:

$$\mathcal L = -\frac1N\sum_t \log\pi_\theta(a_t \mid s_t)\,\hat A_t \;+\; c_v\,\frac1N\sum_t \big(V_\phi(s_t) - \hat R_t\big)^2 \;-\; c_e\,\frac1N\sum_t \mathcal H\big(\pi_\theta(\cdot \mid s_t)\big).$$

- **Policy term.** Minimising $-\log\pi \cdot \hat A$ ascends the policy objective. $\hat A_t$ must be a constant (detached) so no gradient flows into the critic. Normalising advantages per batch (mean 0, sd 1) keeps step sizes stable.
- **Value term.** Regress $V_\phi$ onto $\hat R_t = \hat A_t + V(s_t)$, weight $c_v \approx 0.5$. Actor and critic often share a body with two heads, like AlphaGo Zero's policy-value network (09.03).
- **Entropy bonus.** $\mathcal H(\pi) = -\sum_a \pi(a \mid s)\log\pi(a \mid s)$ is largest when probabilities are spread out (it equals $\ln 4 \approx 1.39$ nats for four equally likely actions) and zero when one action is certain. Subtracting $c_e \mathcal H$ from the loss rewards spread; $c_e \approx 0.01$ is a common default.

The entropy bonus is a dial too. On a one-state problem with four actions (mean rewards 1.0, 0.8, 0.2, 0.0, noise standard deviation 1, an aggressive step size, 200 runs per setting), a small bonus helps and a large one hurts:

~~~dg-fig 08.06-entropy.svg
Left: without a bonus the policy's entropy collapses within a few dozen updates; with c_e = 0.5 it never commits. Right: the share of runs that settle on the best action rises from 0.84 (no bonus) to about 0.90 at c_e = 0.05-0.1, then falls to 0.67 at c_e = 0.5.
~~~

Without the bonus, lucky early samples of the second-best action can make the policy nearly deterministic before the best one is tried enough, and then it stops generating the data that would correct it. With too much bonus, the policy stays random and the expected reward of the final policy drops (0.98 at $c_e = 0.05$ versus 0.77 at $c_e = 0.5$ in this run).

## Actor-critic in code

A complete A2C on the noisy corridor from Step 3, in PyTorch, with a shared body and one-hot states.

```python
import numpy as np, torch, torch.nn as nn
torch.manual_seed(0); torch.set_num_threads(1); rng = np.random.default_rng(0)

# Noisy corridor: start in cell 1, goal in cell 9, reward -1 + N(0, 2^2) per step
N, GAMMA, LAM = 10, 0.95, 0.9
def step(s, a):
    s2 = np.clip(s + (1 if a == 1 else -1), 1, N - 1)
    return s2, -1.0 + 2.0 * rng.normal(), s2 == N - 1

body = nn.Sequential(nn.Linear(N, 32), nn.Tanh())
actor, critic = nn.Linear(32, 2), nn.Linear(32, 1)          # shared body, two heads
params = [*body.parameters(), *actor.parameters(), *critic.parameters()]
opt = torch.optim.Adam(params, lr=3e-3)
onehot = lambda s: torch.eye(N)[torch.as_tensor(s)]

def gae(r, v, v_last, done, gamma=GAMMA, lam=LAM):
    adv, nxt, nv = np.zeros(len(r)), 0.0, (0.0 if done else v_last)
    for t in reversed(range(len(r))):
        delta = r[t] + gamma * nv - v[t]
        nxt = delta + gamma * lam * nxt
        adv[t], nv = nxt, v[t]
    return adv

def rollout(max_steps=100):
    s, S, A, R = 1, [], [], []
    for _ in range(max_steps):
        with torch.no_grad():
            a = torch.distributions.Categorical(logits=actor(body(onehot(s)))).sample().item()
        s2, r, done = step(s, a); S.append(s); A.append(a); R.append(r); s = s2
        if done: break
    return S, A, R, s, done

for it in range(301):
    S, A, R, s_last, done = rollout()
    h = body(onehot(S)); dist = torch.distributions.Categorical(logits=actor(h))
    v = critic(h).squeeze(-1)
    with torch.no_grad(): v_last = critic(body(onehot(s_last))).item()
    adv = torch.tensor(gae(R, v.detach().numpy(), v_last, done), dtype=torch.float32)
    ret = adv + v.detach()                                    # value targets
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)             # detached, normalised
    loss = (-(dist.log_prob(torch.tensor(A)) * adv).mean()    # policy term
            + 0.5 * ((v - ret) ** 2).mean()                   # value term
            - 0.01 * dist.entropy().mean())                   # entropy bonus
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 100 == 0:
        with torch.no_grad():
            p_right = torch.softmax(actor(body(onehot(range(1, N - 1)))), -1)[:, 1]
            V = critic(body(onehot(range(1, N - 1)))).squeeze(-1)
        print(f"iter {it:3d}  steps {len(R):3d}  P(right) {p_right.mean():.2f}  V(start) {V[0]:6.2f}")
# iter   0  steps  95  P(right) 0.55  V(start)  -0.09
# iter 100  steps   8  P(right) 0.96  V(start)  -4.44
# iter 200  steps   8  P(right) 0.97  V(start)  -6.46
# iter 300  steps   8  P(right) 0.98  V(start)  -6.29
```

Eight steps is the shortest path, and the true value of the start under the optimal policy is $-(1 - 0.95^8)/0.05 \approx -6.73$, so both actor and critic end up close. The lines that matter most: `v.detach()` when building targets and advantages, the normalisation, and bootstrapping from `v_last` only when the episode did not truly end. Real implementations run many environments in parallel and update on fixed-length chunks (say 8 environments x 16 steps), which lets A2C learn even when episodes never end.

## Reading the result

Policy-gradient training curves are noisy, so judge them carefully.

~~~dg-fig 08.06-cartpole.svg
CartPole (balance a pole on a cart; +1 per step, at most 500 steps), 5 seeds per method, mean line and min-max band. In this run the learned-baseline version reached about 420 by 100k steps, while our A2C (8 envs x 16 steps, GAE λ = 0.95) wandered between 100 and 240 and plain REINFORCE plateaued near 200.
~~~

Each method used the better of two learning rates from a one-seed check, Adam, 64-unit tanh networks, $c_e = 0.01$ and (with a critic) normalised advantages. What to read from it:

- **Actor-critic is not automatically better.** CartPole's reward is a noise-free +1 per step, so Monte Carlo returns are not very noisy and a normalised baseline removes most of the variance. A2C's bootstrapped targets lean on a critic that is poor early on, and that bias showed up as slower, less stable progress. Where per-step rewards are noisy, as in the corridor, the critic's variance reduction matters far more (see the variance figure).
- **Look at seeds, not just the mean.** A2C's final lengths over five seeds were 272, 103, 343, 107 and 106: two seeds learned well, three stalled. A single-seed curve would have told either story.
- **Track diagnostics, not only return:** entropy (should fall gradually, not crash), value loss, and the critic's **explained variance** $1 - \text{Var}(\hat R - V)/\text{Var}(\hat R)$: near 1 the critic predicts well; near 0 or below, its advantages are mostly noise.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Policy "chases" the critic, training unstable | advantage not detached, so the policy term trains the critic | build advantages from `v.detach()`; use live `v` only in the value loss
| Entropy collapses in the first few hundred updates | step size too large, no entropy bonus | lower the learning rate, raise $c_e$, normalise advantages
| Value loss dominates, policy barely moves | rewards on a large scale, shared body | normalise rewards or returns, lower $c_v$, or split the networks
| Learning stalls on long-horizon tasks | $\lambda$ or $\gamma$ too small, critic biased | raise $\lambda$; check bootstrapping at truncation
| Agent learns that time-outs are "deaths" | treating a time-limit cut as a terminal state | at truncation bootstrap from $V(s_T)$; only true terminals use 0

On the time-limit row: a CartPole episode cut at 500 steps did not fail; using 0 there teaches the critic that its best states are worthless.

## Going deeper

**Where the theorem comes from.** With $J(\theta) = \sum_\tau P_\theta(\tau) R(\tau)$ over trajectories $\tau$, $\nabla J = \sum_\tau P_\theta(\tau)\,\nabla \log P_\theta(\tau)\,R(\tau)$ (the log-derivative trick, $\nabla P = P\,\nabla\log P$). The log-probability of a trajectory is a sum of environment terms, which do not depend on $\theta$, and policy terms $\log \pi_\theta(a_t \mid s_t)$; only the latter survive the gradient. Causality then lets you drop rewards that came before each action, giving $G_t$; the baseline argument from Step 2 lets you subtract $V(s_t)$.

**On-policy.** The expectation is over the current policy's trajectories. After one gradient step the batch was generated by an older policy, so A2C uses each batch once. **PPO** reuses a batch for several epochs by weighting with the importance ratio $\pi_\theta / \pi_{\text{old}}$ and clipping it to $[1-\epsilon, 1+\epsilon]$. **GRPO**, used for training LLMs on reasoning tasks, removes the critic and computes advantages relative to a group of samples for the same prompt. Both are covered in the AI Research course's Reinforcement Learning chapter, and RLHF with PPO is the subject of the next lesson.

**Continuous actions.** With a Gaussian policy $\log\pi$ has a closed form and everything above carries over. Off-policy actor-critics (DDPG, TD3, SAC) instead learn a Q-function critic and differentiate through it.

**A2C vs A3C.** A3C ran many learners asynchronously against shared parameters; A2C averages their gradients into one synchronous update, which is simpler and deterministic.

## Key terms

| Term | Meaning
| --- | --- |
| Policy $\pi_\theta(a \mid s)$ | The actor: a parameterised probability distribution over actions
| Policy gradient | $\mathbb E[\nabla\log\pi_\theta(a \mid s)\,\Psi]$, the direction that raises expected return
| Baseline | A state-only quantity subtracted from the return; cuts variance without adding bias
| Advantage $A(s,a)$ | How much better action $a$ is than the policy's average from $s$: $Q(s,a) - V(s)$
| Critic $V_\phi(s)$ | A learned value function that grades the actor's choices
| TD error $\delta_t$ | $r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$, the one-step advantage estimate
| GAE | $\sum_l (\gamma\lambda)^l \delta_{t+l}$; $\lambda$ moves from TD ($\lambda=0$) to Monte Carlo ($\lambda=1$)
| Entropy bonus | A reward for keeping the policy spread out, which delays premature commitment
| On-policy | Gradients are valid only for data from the current policy

<div class="take">
<div class="lab">Key takeaway</div>

an actor-critic follows the policy gradient, weighting each action's log-probability by an advantage from a learned critic; GAE's $\lambda$ sets how far to trust that critic (bias) versus the raw returns (variance), and the A2C loss adds value regression and an entropy bonus, with the advantage always detached from the critic.

</div>
