---
id: "08.02"
course: "deep-learning"
module: "08"
slug: "dynamic-programming"
title: "Dynamic programming"
keyIdeas: ["Dynamic programming solves a known MDP by repeatedly applying Bellman backups; it needs $P$ and $R$ and costs $O(|\\mathcal S|^2|\\mathcal A|)$ per sweep.", "Iterative policy evaluation applies the Bellman expectation backup until values stop changing; it converges because the backup is a $\\gamma$-contraction in the max norm.", "Policy improvement makes the policy greedy with respect to $Q^\\pi$; the policy improvement theorem guarantees the new policy is at least as good everywhere.", "Policy iteration alternates full evaluation and improvement and usually converges in a handful of iterations; value iteration folds them into one max-backup per sweep.", "Error after $k$ value-iteration sweeps shrinks like $\\gamma^k$, so $\\gamma=0.99$ needs roughly ten times more sweeps than $\\gamma=0.9$."]
question: "On a 10,000-state gridworld with $\\gamma=0.99$, value iteration takes about 1,800 sweeps to converge, while policy iteration converges in 6 iterations. Which is faster in wall-clock time, and what does the answer depend on? Suggest a middle ground."
answer: "It depends on the cost of each policy-iteration evaluation. If evaluation is done exactly by solving the linear system, each iteration costs $O(|S|^3)\\approx 10^{12}$ operations for a dense matrix (much less if $P_\\pi$ is sparse, as in a gridworld where each state has a few successors and a sparse solver works well), versus about $|S|\\cdot|A|\\cdot$(successors) per value-iteration sweep. With sparse transitions, 6 sparse solves are often cheaper than 1,800 sweeps; with dense transitions, value iteration wins. If evaluation is iterative, each policy-iteration step itself needs hundreds of sweeps at $\\gamma=0.99$, so the total sweep counts become comparable. The middle ground is modified policy iteration: run a fixed small number $m$ of evaluation sweeps (say 10 to 50) between greedy improvements. It keeps policy iteration's fast policy convergence without paying for exact evaluation of early, poor policies."
sources: [{"title": "Sutton and Barto: Reinforcement Learning, An Introduction (2nd ed.)", "url": "https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf", "note": "Chapter 4: policy evaluation, the improvement theorem, policy and value iteration, GPI."}, {"title": "David Silver: UCL Course on RL", "url": "https://www.davidsilver.uk/teaching/", "note": "Lecture 3 (planning by DP), including the contraction-mapping proof."}]
practice: ["value-iteration", "policy-iteration", "value-iteration-step", "bellman-expectation-equation"]
tags: ["dynamic programming", "value iteration", "policy iteration"]
tools: ["NumPy"]
---

## Planning with a known model

Lesson 08.01 left two equations: Bellman expectation (linear, for a fixed policy) and Bellman optimality (with a max). **Dynamic programming** (DP) solves them when you know the model $P(s'\mid s,a)$ and $R$, for example a gridworld, an inventory problem with known demand, or a board game's rules. DP is not learning from experience, but every RL algorithm in this module is an approximation of a DP update with samples in place of the model, so the structure is worth knowing exactly.

The basic operation is a **Bellman backup**: replace a state's value with the right-hand side of a Bellman equation using current estimates of its successors.

## Iterative policy evaluation

Instead of inverting $(I-\gamma P_\pi)$, start from any $V_0$ and sweep:

$$V_{k+1}(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)\big[R(s,a,s')+\gamma V_k(s')\big]$$

Why does it converge? The backup operator $T^\pi$ is a **$\gamma$-contraction** in the max norm: $\|T^\pi U-T^\pi V\|_\infty\le\gamma\|U-V\|_\infty$, because the only place $V$ enters is multiplied by $\gamma$ inside an average. By the Banach fixed-point theorem it has a unique fixed point $V^\pi$, and the error shrinks by at least a factor $\gamma$ per sweep. Updating values in place (using new values as soon as they are computed) usually converges faster and is equally valid.

## Policy improvement and policy iteration

Given $V^\pi$, compute $Q^\pi(s,a)=\sum_{s'}P(s'\mid s,a)[R+\gamma V^\pi(s')]$ and act greedily: $\pi'(s)=\arg\max_aQ^\pi(s,a)$. The **policy improvement theorem** says $V^{\pi'}(s)\ge V^\pi(s)$ for every state: taking a better first action and then following $\pi$ is already at least as good, and following $\pi'$ thereafter only helps more. If nothing improves, $\pi$ satisfies the optimality equation and is optimal.

**Policy iteration** alternates the two to convergence:

~~~dg-flow Policy iteration: evaluate the current policy exactly, then make it greedy; stop when it no longer changes.
Policy pi -(evaluate)-> V pi -(greedy w.r.t. Q)-> *Improved pi -(repeat until stable)-> Optimal pi
~~~

Because there are finitely many deterministic policies and each iteration strictly improves until it stops, policy iteration terminates in a finite number of iterations, in practice very few (often under 10 even for thousands of states), though each iteration contains a full evaluation.

## Value iteration

Policy evaluation does not need to converge before improving. Truncate it to a single sweep and fold the greedy step into the backup, and you get **value iteration**:

$$V_{k+1}(s)=\max_a\sum_{s'}P(s'\mid s,a)\big[R(s,a,s')+\gamma V_k(s')\big]$$

This is the Bellman optimality backup, also a $\gamma$-contraction, so $\|V_k-V^*\|_\infty\le\gamma^k\|V_0-V^*\|_\infty$. Stop when the largest change in a sweep falls below a threshold $\theta$, then extract the greedy policy.

```python
import numpy as np
def value_iteration(P, R, gamma, tol=1e-8):     # P: (S, A, S), R: (S, A)
    V = np.zeros(P.shape[0])
    while True:
        Q = R + gamma * P @ V                    # (S, A): one-step lookahead
        V_new = Q.max(1)
        if np.abs(V_new - V).max() < tol:
            return V_new, Q.argmax(1)
        V = V_new
```

<div class="ex">
<div class="lab">Worked example</div>

**how many sweeps?**

Initial error $\|V_0-V^*\|_\infty=10$ and you want error below $10^{-3}$, a factor of $10^{-4}$.

With $\gamma=0.9$: $k\ge\ln(10^{-4})/\ln(0.9)=87.4$, so 88 sweeps.

With $\gamma=0.99$: $k\ge\ln(10^{-4})/\ln(0.99)=916$ sweeps. A longer horizon makes every DP and TD method slower to propagate value, for the same reason.

</div>

## Generalised policy iteration and the limits of DP

Policy iteration, value iteration and everything in between (a few evaluation sweeps per improvement, asynchronous updates of single states) are instances of **generalised policy iteration**: values are pushed towards the current policy's values, the policy is pushed towards greedy, and the two settle together at optimality. The same pattern reappears in Q-learning (08.04), actor-critic (08.06) and, strikingly, AlphaGo Zero, where search improves the policy and training evaluates it (09.04).

| Method | Backup | Per iteration | Iterations
| --- | --- | --- | --- |
| Policy evaluation | expectation, fixed $\pi$ | $O(\lvert S\rvert^2\lvert A\rvert)$ per sweep | until $\gamma^k$ small
| Policy iteration | evaluation to convergence, then greedy | many sweeps | few
| Value iteration | optimality (max) | one sweep | about $\log(\epsilon)/\log\gamma$

DP has two hard limits. It needs the model, which you usually do not have; the next lessons replace the sum over $s'$ with sampled transitions. And it sweeps every state, which is impossible when states are Go positions (about $10^{170}$) or camera images; that motivates function approximation (deep RL, covered in AI Research) and focusing computation on states that matter (tree search, Module 09).

<div class="take">
<div class="lab">Key takeaway</div>

with a known model, Bellman backups are contractions, so repeated sweeps converge; policy iteration alternates exact evaluation with greedy improvement, value iteration does one max-backup per sweep, and both are the template that sample-based RL approximates.

</div>
