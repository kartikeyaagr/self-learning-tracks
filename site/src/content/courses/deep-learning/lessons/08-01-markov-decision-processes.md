---
id: "08.01"
course: "deep-learning"
module: "08"
slug: "markov-decision-processes"
title: "Markov decision processes"
keyIdeas: ["An MDP is $(\\mathcal S,\\mathcal A,P,R,\\gamma)$: states, actions, transition probabilities $P(s'\\mid s,a)$, rewards and a discount. The Markov property says the current state is all you need to predict what comes next.", "The agent maximises the expected return $G_t=\\sum_{k\\ge0}\\gamma^k r_{t+k+1}$; $\\gamma$ sets an effective horizon of about $1/(1-\\gamma)$ steps.", "$V^\\pi(s)$ and $Q^\\pi(s,a)$ are expected returns under a policy $\\pi$; they satisfy the Bellman expectation equation, a linear system you can solve exactly when the model is known.", "The optimal values satisfy the Bellman optimality equation, with a max over actions; acting greedily with respect to $Q^*$ is optimal.", "Most RL bugs at this level are bookkeeping: bootstrapping across episode ends, mixing up $r_t$ and $r_{t+1}$, or treating a time-limit truncation as a true terminal."]
question: "A robot task has a 500-step time limit. Your value-learning code treats step 500 as terminal, so the target there is just $r$. Training converges, but the robot behaves badly near the end of each episode and values for identical states differ depending on the time step. Explain the bug using the definition of return and the Markov property, and give the fix."
answer: "The state the agent observes does not include time, so under the Markov assumption a given state should have one value regardless of when it is reached. Treating the time-limit cutoff as terminal tells the learner that the return from that state is just the last reward, while the same state reached earlier has a long future. Values for identical states then conflict and the learned values near the cutoff are biased low; the policy acts as if the world ends, for example taking risky actions or stopping effort. The cutoff is a truncation, not a property of the environment. Fix: keep separate `terminated` and `truncated` flags and bootstrap with $r+\\gamma V(s')$ on truncation (only true terminals use $r$ alone). Alternatively, if the time limit really is part of the task, add the remaining time to the state so the problem is Markov again."
sources: [{"title": "Sutton and Barto: Reinforcement Learning, An Introduction (2nd ed.)", "url": "https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf", "note": "Chapter 3: MDPs, returns, value functions and both Bellman equations."}, {"title": "OpenAI Spinning Up: Key Concepts in RL", "url": "https://spinningup.openai.com/en/latest/spinningup/rl_intro.html", "note": "Compact notation for policies, returns, value functions and Bellman equations."}, {"title": "David Silver: UCL Course on RL", "url": "https://www.davidsilver.uk/teaching/", "note": "Lecture 2 (MDPs) slides and video; the matrix form of policy evaluation."}]
practice: ["discounted-returns", "bellman-expectation-equation", "value-iteration-step"]
tags: ["MDP", "Bellman equation", "return"]
tools: ["NumPy"]
---

## Where this module sits

This module is the classical core of reinforcement learning: the problem definition (this lesson), exact solutions when the environment is known (08.02), learning values from experience (08.03, 08.04), exploration (08.05), policy-based learning (08.06) and learning rewards from human preferences (08.07). The AI Research course's Reinforcement Learning chapter introduces agents and environments and then goes deep on DQN, PPO and GRPO; this module gives the tabular foundations those methods approximate, and Module 09 uses them to build AlphaGo Zero.

## The interaction loop

An agent observes a state $s_t$, picks an action $a_t$, and the environment returns a reward $r_{t+1}$ and a next state $s_{t+1}$. A **Markov decision process** formalises this as a tuple $(\mathcal S,\mathcal A,P,R,\gamma)$:

- $\mathcal S$, $\mathcal A$: states and actions (finite in this module; continuous later).
- $P(s'\mid s,a)$: transition probabilities. The **Markov property** says this depends only on the current $s$ and $a$, not on earlier history. If it does not hold, enlarge the state until it does: stacking the last 4 Atari frames to capture velocity, or 8 past Go positions to detect repetition (lesson 09.01).
- $R(s,a)$ or $R(s,a,s')$: expected immediate reward.
- $\gamma\in[0,1)$: discount factor.

~~~dg-flow The agent-environment loop; the environment is a black box that samples from P and R.
Agent -(action a)-> *Environment -(reward r, next state)-> Agent
~~~

A **policy** $\pi(a\mid s)$ maps states to action probabilities. Deterministic policies are the special case with all mass on one action.

## Returns and discounting

The agent's objective is the **return**, the discounted sum of future rewards:

$$G_t=r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3}+\dots=r_{t+1}+\gamma G_{t+1}$$

The recursive form is the one you compute: walk backwards through an episode. Discounting keeps infinite sums finite in continuing tasks, encodes a preference for sooner rewards, and gives an effective horizon of about $1/(1-\gamma)$ steps: 10 for $\gamma=0.9$, 100 for 0.99. Episodic tasks end at a **terminal state**, after which all rewards are zero, so $G$ must not bootstrap past the end. A time limit is different: the state was not terminal, the episode was just cut, and value estimates should still bootstrap there. Mixing the two up silently biases values.

<div class="ex">
<div class="lab">Worked example</div>

**returns for one episode**

Rewards $r_1, r_2, r_3=1, 0, 2$, then the episode terminates; $\gamma=0.9$.

Backwards: $G_2=2$; $G_1=0+0.9\cdot2=1.8$; $G_0=1+0.9\cdot1.8=2.62$.

With a batch holding several episodes back to back, the recursion must reset at each `done` flag, or rewards from the next episode leak into this one.

</div>

## Value functions and the Bellman expectation equation

The **state value** $V^\pi(s)=\mathbb E_\pi[G_t\mid s_t=s]$ is the expected return from $s$ when following $\pi$. The **action value** $Q^\pi(s,a)=\mathbb E_\pi[G_t\mid s_t=s,a_t=a]$ fixes the first action. They relate by $V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a)$. Expanding one step of the return gives the **Bellman expectation equation**:

$$V^\pi(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)\big[R(s,a,s')+\gamma V^\pi(s')\big]$$

For a fixed policy, this is linear. Stack values into a vector, let $P_\pi$ be the state-to-state transition matrix under $\pi$ and $r_\pi$ the expected one-step reward; then $V=r_\pi+\gamma P_\pi V$, so

$$V^\pi=(I-\gamma P_\pi)^{-1}r_\pi$$

The inverse exists for $\gamma<1$. This is exact policy evaluation, practical for a few thousand states ($O(|\mathcal S|^3)$).

```python
import numpy as np
def evaluate(P, R, pi, gamma):          # P: (S, A, S), R: (S, A), pi: (S, A)
    P_pi = np.einsum('sa,sat->st', pi, P)
    r_pi = (pi * R).sum(1)
    return np.linalg.solve(np.eye(len(r_pi)) - gamma * P_pi, r_pi)
```

<div class="ex">
<div class="lab">Worked example</div>

**a two-state loop**

From A, the only action gives reward 1 and moves to B; from B, reward 0 and back to A; $\gamma=0.9$.

$V(A)=1+0.9V(B)$ and $V(B)=0.9V(A)$, so $V(A)=1+0.81V(A)$, giving $V(A)=1/0.19=5.26$ and $V(B)=4.74$.

A collects 1 every second step forever: $1+0.81+0.81^2+\dots=5.26$, as the closed form says.

</div>

## Optimality

An optimal policy has the highest value in every state at once, and for finite MDPs one always exists that is deterministic. Its values $V^*$, $Q^*$ satisfy the **Bellman optimality equation**:

$$Q^*(s,a)=\sum_{s'}P(s'\mid s,a)\big[R(s,a,s')+\gamma\max_{a'}Q^*(s',a')\big]$$

The max makes it non-linear, so there is no single matrix solve; the next lesson's iterative methods handle it. Once you have $Q^*$, acting is trivial: $\pi^*(s)=\arg\max_a Q^*(s,a)$. This is why so much of RL is about estimating $Q$: a good $Q$ is a policy, without needing a model of $P$.

| Object | Definition | Needs a model to act?
| --- | --- | --- |
| $V^\pi(s)$ | expected return from $s$ following $\pi$ | yes, to look one step ahead
| $Q^\pi(s,a)$ | expected return after taking $a$ in $s$, then $\pi$ | no
| $V^*$, $Q^*$ | best achievable values | $Q^*$ no, $V^*$ yes
| Advantage $A^\pi(s,a)$ | $Q^\pi(s,a)-V^\pi(s)$ | used by actor-critic (08.06)

<div class="take">
<div class="lab">Key takeaway</div>

an MDP is states, actions, Markov transitions, rewards and a discount; values are expected discounted returns, the Bellman expectation equation is linear and solvable exactly for a fixed policy, and the Bellman optimality equation adds a max whose greedy solution is the optimal policy.

</div>
