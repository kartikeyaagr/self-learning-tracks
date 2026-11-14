---
id: "08.01"
course: "deep-learning"
module: "08"
slug: "markov-decision-processes"
title: "Markov decision processes"
keyIdeas: ["An MDP is $(\\mathcal S,\\mathcal A,P,R,\\gamma)$: states, actions, transition probabilities $P(s'\\mid s,a)$, rewards and a discount. The **Markov property** says the current state is all you need to predict what comes next; if it fails, add history to the state.", "The agent maximises the expected **return** $G_t=\\sum_{k\\ge0}\\gamma^k r_{t+k+1}=r_{t+1}+\\gamma G_{t+1}$, computed backwards through an episode; $\\gamma$ sets an effective horizon of about $1/(1-\\gamma)$ steps and can change which policy is optimal.", "$V^\\pi(s)$ and $Q^\\pi(s,a)$ are expected returns under a policy $\\pi$; they satisfy the **Bellman expectation equation**, a linear system $V^\\pi=(I-\\gamma P_\\pi)^{-1}r_\\pi$ you can solve exactly when the model is known.", "The optimal values satisfy the **Bellman optimality equation**, with a max over actions; acting greedily with respect to $Q^*$ is optimal, so a good $Q$ is already a policy.", "Most RL bugs at this level are bookkeeping: bootstrapping across episode ends, mixing up $r_t$ and $r_{t+1}$, or treating a time-limit truncation as a true terminal."]
question: "A robot task has a 500-step time limit. Your value-learning code treats step 500 as terminal, so the target there is just $r$. Training converges, but the robot behaves badly near the end of each episode and values for identical states differ depending on the time step. Explain the bug using the definition of return and the Markov property, and give the fix."
answer: "The state the agent observes does not include time, so under the Markov assumption a given state should have one value regardless of when it is reached. The return is defined as the discounted sum of all future rewards, and reaching step 500 does not end the robot's world; it only ends the data collection. Treating the cutoff as terminal tells the learner that the return from that state is just the last reward, while the same state reached earlier has a long future. Values for identical states then receive conflicting targets: the learned values are an average over these, biased low overall and especially for states typically seen near the cutoff. (In a toy version with reward 1 per step, $\\gamma=0.9$ and a 20-step limit, the bug pulls the learned value from the true 10 down to about 6.9.) The policy then acts as if the world ends, for example taking risky actions or stopping effort late in the episode. The cutoff is a truncation, not a property of the environment. Fix: keep separate `terminated` and `truncated` flags (as Gymnasium does) and bootstrap with $r+\\gamma V(s')$ on truncation; only true terminals use $r$ alone, and the backward return recursion should multiply by $(1-\\text{terminated})$, not by $(1-\\text{done})$. Alternatively, if the time limit really is part of the task, add the remaining time to the state so the problem is Markov again and different time steps legitimately get different values."
sources: [{"title": "Sutton and Barto: Reinforcement Learning, An Introduction (2nd ed.)", "url": "https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf", "note": "Chapter 3: MDPs, returns, value functions and both Bellman equations."}, {"title": "OpenAI Spinning Up: Key Concepts in RL", "url": "https://spinningup.openai.com/en/latest/spinningup/rl_intro.html", "note": "Compact notation for policies, returns, value functions and Bellman equations."}, {"title": "David Silver: UCL Course on RL", "url": "https://www.davidsilver.uk/teaching/", "note": "Lecture 2 (MDPs) slides and video; the matrix form of policy evaluation."}]
practice: ["discounted-returns", "bellman-expectation-equation", "value-iteration-step"]
tags: ["MDP", "Bellman equation", "return"]
tools: ["NumPy"]
---

## What you will learn

- What a **Markov decision process** (MDP) is: the five ingredients that turn "an agent acting in a world" into a precise maths problem.
- What the Markov property demands, and how to repair a state that breaks it.
- How the discounted **return** is computed, what the discount $\gamma$ does, and why a time limit is not the same as an ending.
- What the value functions $V^\pi$ and $Q^\pi$ mean, and how the **Bellman expectation equation** lets you compute them exactly with one linear solve.
- What "optimal" means, how the **Bellman optimality equation** differs, and why a good $Q$ is already a policy.

*Before you start:* you need expected values (a probability-weighted average), basic matrix notation and `np.linalg.solve`. Nothing from earlier deep-learning modules is required; this lesson is self-contained.

## Where this module starts

This module is the classical core of reinforcement learning (RL): the problem definition (this lesson), exact solutions when the environment is known (08.02), learning values from experience (08.03, 08.04), exploration (08.05), policy-based learning (08.06) and learning rewards from human preferences (08.07). The AI Research course's Reinforcement Learning chapter introduces agents and environments and then goes deep on DQN, PPO and GRPO; this module gives the tabular foundations those methods approximate, and Module 09 uses them to build AlphaGo Zero.

Unlike supervised learning, RL is never told the right answer: the agent tries actions, receives rewards, and must work out which choices earned them. The MDP is the precise description of that world, and every algorithm in this module solves one.

## The idea in one picture

A small robot stands in a corridor of five cells. At the left end lies a coin worth **+1**; at the right end a prize worth **+10**. Reaching either end finishes the episode. In each cell the robot picks **left** or **right**, but the floor is slippery: the robot moves the way it chose with probability 0.8 and slides the other way with probability 0.2.

~~~dg-fig 08.01-corridor.svg
The corridor MDP. Under "always go right" with discount 0.9, values rise towards the prize; under a coin-flip policy every value is lower, because the robot wanders and often ends on the +1 coin.
~~~

The numbers under each cell answer one question: "if the robot starts here and follows this rule forever, how much discounted reward does it collect on average?" That average is the **value** of the cell under that rule (the **policy**). Read the figure:

- Under "always right", cell 5 is worth 9.44: it usually steps straight onto the prize, and less than the full 10 because of slips and discounting.
- Cell 1 is worth only 4.28 even under the same policy: the prize is five steps away, each step shrinks future reward by 0.9, and slips can dump the robot on the coin.
- The random policy scores lower everywhere (1.35 to 6.99). Same world, worse behaviour, lower values.

This lesson shows where those numbers come from and how they tell you which action is best.

## Step 1: the five ingredients

At each time step $t$ the agent observes a **state** $s_t$, picks an **action** $a_t$, and the environment replies with a **reward** $r_{t+1}$ and a **next state** $s_{t+1}$. Then it repeats.

~~~dg-flow The agent-environment loop; the environment is a black box that samples from P and R.
Agent -(action a)-> *Environment -(reward r, next state)-> Agent
~~~

A Markov decision process writes that loop as a tuple $(\mathcal S,\mathcal A,P,R,\gamma)$:

- $\mathcal S$: the set of **states**. Corridor: cells 0 to 6, where 0 and 6 are terminal.
- $\mathcal A$: the set of **actions**. Corridor: {left, right}. Finite in this module; continuous later.
- $P(s'\mid s,a)$: the **transition probabilities**, the chance of landing in $s'$ after doing $a$ in $s$. Corridor: $P(4\mid 3,\text{right})=0.8$ and $P(2\mid 3,\text{right})=0.2$.
- $R(s,a)$ or $R(s,a,s')$: the expected immediate **reward**. Corridor: 10 for entering cell 6, 1 for entering cell 0, otherwise 0.
- $\gamma\in[0,1)$: the **discount factor**, how much less a reward one step later is worth. We use 0.9.

Timing convention: the reward for acting at time $t$ is $r_{t+1}$, because it arrives with the next state. Keep this convention fixed.

For a finite MDP, $P$ is just an array of shape $(|\mathcal S|, |\mathcal A|, |\mathcal S|)$ whose last axis sums to 1. Here is the slice for cell 3:

| From cell 3, action | to cell 2 | to cell 4 | expected reward
| --- | --- | --- | --- |
| left | 0.8 | 0.2 | 0
| right | 0.2 | 0.8 | 0

## Step 2: the Markov property

The **Markov** assumption: the next state and reward depend only on the current state and action, not on the path that led there.

$$P(s_{t+1}\mid s_t,a_t,s_{t-1},a_{t-1},\dots)=P(s_{t+1}\mid s_t,a_t).$$

The state is a complete summary of the past for predicting the future. In the corridor this holds by construction.

When it fails, **enlarge the state until it holds**. One Atari frame shows where the ball is but not where it is going, so DQN stacks the last 4 frames to capture velocity. One Go position cannot tell whether a move would repeat an earlier position, so AlphaGo Zero feeds the last 8 positions (lesson 09.01). If the task has a deadline, the remaining time must be part of the state; we will see what happens when it is not.

## Step 3: policies

A **policy** $\pi(a\mid s)$ is the agent's rule for choosing actions: for every state, a probability for each action. "Always right" puts probability 1 on right in every cell; the random policy puts 0.5 on each. A **deterministic** policy is the special case with all mass on one action, written $\pi(s)$.In code it is an array of shape $(|\mathcal S|, |\mathcal A|)$ with rows summing to 1.

## Step 4: returns and discounting

The agent wants a big total, not just a big next reward. The **return** from time $t$ is the discounted sum of all future rewards:

$$G_t=r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3}+\dots=\sum_{k\ge 0}\gamma^k r_{t+k+1}=r_{t+1}+\gamma G_{t+1}.$$

The reward $k$ steps ahead is multiplied by $\gamma^k$. The recursive form is the one you compute, walking backwards from the end of an episode.

~~~dg-fig 08.01-discount.svg
How much a reward k steps ahead counts. The weight falls to about 37% after 1/(1-γ) steps: roughly 2 steps for γ = 0.5, 10 for 0.9 and 100 for 0.99.
~~~

Discounting does three jobs:

- It keeps sums finite in **continuing tasks** that never end: $|G|\le r_{\max}/(1-\gamma)$.
- It prefers sooner rewards.
- It sets an **effective horizon** of about $1/(1-\gamma)$ steps: 10 for $\gamma=0.9$, 100 for 0.99. Rewards much further out barely matter.

<div class="ex">
<div class="lab">Worked example</div>

**returns for one episode**

Rewards $r_1, r_2, r_3 = 1, 0, 2$, then the episode terminates; $\gamma=0.9$.

Start at the end: $G_2 = r_3 = 2$ (nothing comes after a terminal state).

One step back: $G_1 = r_2 + 0.9\,G_2 = 0 + 0.9 \times 2 = 1.8$.

One more: $G_0 = r_1 + 0.9\,G_1 = 1 + 0.9 \times 1.8 = 1 + 1.62 = 2.62$.

Check by the direct sum: $1 + 0.9 \times 0 + 0.81 \times 2 = 2.62$. Same answer.

With a batch holding several episodes back to back, the recursion must reset at each `done` flag, or rewards from the next episode leak into this one.

</div>

**Terminal versus truncated.** Episodic tasks end at a **terminal state** (the coin or the prize). After it all rewards are zero, so $G$ must not look past it. A **time limit** is different: the state was not terminal, the run was merely cut off for convenience. Value estimates should still look ahead there ("bootstrap" with $\gamma V(s')$). Gymnasium reports the two separately as `terminated` and `truncated`; mixing them up silently biases values, as the "What goes wrong" section shows.

## Step 5: value functions

A single return is random, so values are defined as averages.

- The **state value** $V^\pi(s)=\mathbb E_\pi[G_t\mid s_t=s]$: the expected return from $s$ when following $\pi$.
- The **action value** $Q^\pi(s,a)=\mathbb E_\pi[G_t\mid s_t=s,a_t=a]$: the expected return if you take $a$ first, then follow $\pi$.

They are linked by $V^\pi(s)=\sum_a\pi(a\mid s)\,Q^\pi(s,a)$: the value of a state is the policy-weighted average of its action values.

You can estimate a value by brute force: start many episodes in the cell and average their returns.

~~~dg-fig 08.01-mc.svg
Left: three independent runs of 2000 episodes from cell 3 (always right). Their running averages settle on 6.80, the value the Bellman equation gives. Right: individual returns range from below 1 to 8.1, which is why we need the average.
~~~

In our run, 52% of episodes from cell 3 moved right three times in a row and earned $0.9^2\times 10 = 8.1$; about 2% scored below 2, mostly by sliding onto the coin. The average was 6.82 against the exact 6.80: averaging works but is slow and noisy, while the Bellman equation gives the exact number directly.

## Step 6: the Bellman expectation equation

Split the return into "the first reward" plus "everything after": $G_t=r_{t+1}+\gamma G_{t+1}$. Take expectations and the "everything after" part becomes the value of wherever you land. This gives the **Bellman expectation equation**:

$$V^\pi(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)\big[R(s,a,s')+\gamma V^\pi(s')\big].$$

Read it inside out: for each action (weighted by the policy) and each landing cell (weighted by its probability), add the reward and the discounted value of the landing cell.

<div class="ex">
<div class="lab">Worked example</div>

**check the corridor values by hand**

Policy: always right, $\gamma=0.9$. Terminal cells have value 0.

Cell 5: with prob 0.8 land on the prize (reward 10, then nothing); with prob 0.2 slide to cell 4 (reward 0, then $V(4)$).

$V(5) = 0.8\,(10 + 0) + 0.2\,(0 + 0.9 \times 8.023) = 8 + 0.2 \times 7.221 = 8 + 1.444 = 9.444$.

Cell 4: $V(4) = 0.8 \times 0.9 \times V(5) + 0.2 \times 0.9 \times V(3) = 0.72 \times 9.444 + 0.18 \times 6.796 = 6.800 + 1.223 = 8.023$.

Cell 1: right lands on cell 2 with prob 0.8; a slip lands on the coin (reward 1). $V(1) = 0.8 \times 0.9 \times 5.663 + 0.2 \times 1 = 4.077 + 0.2 = 4.277$.

Every value in the figure satisfies its own equation; that is what evaluating a policy means.

</div>

**One linear solve.** For a fixed policy the equation is linear in the unknown values. Stack the values into a vector $V$. Let $P_\pi$ be the state-to-state matrix under $\pi$, with $P_\pi[s,s']=\sum_a\pi(a\mid s)P(s'\mid s,a)$, and $r_\pi$ the vector of expected one-step rewards. Then $V=r_\pi+\gamma P_\pi V$, so

$$V^\pi=(I-\gamma P_\pi)^{-1}r_\pi.$$

The inverse exists for $\gamma<1$. This is exact **policy evaluation**, practical for a few thousand states since it costs $O(|\mathcal S|^3)$. For bigger problems, the iterative methods of 08.02 and the sampled methods of 08.03 take over.

<div class="ex">
<div class="lab">Worked example</div>

**a two-state loop**

From A, the only action gives reward 1 and moves to B; from B, reward 0 and back to A; $\gamma=0.9$.

Bellman: $V(A)=1+0.9\,V(B)$ and $V(B)=0+0.9\,V(A)$.

Substitute: $V(A)=1+0.81\,V(A)$, so $0.19\,V(A)=1$ and $V(A)=1/0.19=5.26$; then $V(B)=0.9 \times 5.26=4.74$.

Sanity check: A collects 1 every second step forever, so $V(A)=1+0.81+0.81^2+\dots=1/(1-0.81)=5.26$, as the closed form says.

</div>

## Step 7: optimal values and optimal policies

One policy is **better** than another if its value is at least as high in every state. An **optimal policy** $\pi^*$ has the highest value in every state at once. For finite MDPs one always exists, and there is always a deterministic one. Its values $V^*$ and $Q^*$ satisfy the **Bellman optimality equation**:

$$Q^*(s,a)=\sum_{s'}P(s'\mid s,a)\big[R(s,a,s')+\gamma\max_{a'}Q^*(s',a')\big].$$

The policy average is replaced by a **max**: after the first step, act as well as possible. The max makes it non-linear, so there is no single matrix solve; the next lesson's iterative methods handle it.

Once you have $Q^*$, acting is trivial: $\pi^*(s)=\arg\max_a Q^*(s,a)$. No model, no look-ahead, just pick the biggest number. That is why so much of RL is about estimating $Q$: a good $Q$ is a policy.

What is optimal depends on $\gamma$. A short-sighted robot standing next to the coin should grab it; a patient one should walk to the prize.

~~~dg-fig 08.01-gamma.svg
Optimal action per cell, found by value iteration for each γ. Cell 1 takes the coin until γ ≈ 0.65 and cell 2 until γ ≈ 0.39; cells 3 to 5 always head for the +10.
~~~

So $\gamma$ is part of the problem statement, not a harmless knob.

| Object | Definition | Needs a model to act?
| --- | --- | --- |
| $V^\pi(s)$ | expected return from $s$ following $\pi$ | yes, to look one step ahead
| $Q^\pi(s,a)$ | expected return after taking $a$ in $s$, then $\pi$ | no
| $V^*$, $Q^*$ | best achievable values | $Q^*$ no, $V^*$ yes
| Advantage $A^\pi(s,a)$ | $Q^\pi(s,a)-V^\pi(s)$ | used by actor-critic (08.06)

## The corridor MDP in code

A finite MDP is two arrays: `P` of shape (S, A, S) and `R` of shape (S, A). Policy evaluation is one `einsum` and one `solve`.

```python
import numpy as np

# Corridor: cells 0..6; 0 (+1 coin) and 6 (+10 prize) are terminal. Actions: 0 = left, 1 = right.
S, A, slip = 7, 2, 0.2
P = np.zeros((S, A, S)); R = np.zeros((S, A))
for s in range(1, 6):
    for a, d in ((0, -1), (1, +1)):
        for p, step in ((1 - slip, d), (slip, -d)):        # intended move, or slip the other way
            nxt = s + step
            P[s, a, nxt] += p
            R[s, a] += p * {0: 1.0, 6: 10.0}.get(nxt, 0.0)  # expected reward R(s, a)
P[0, :, 0] = P[6, :, 6] = 1.0                           # terminals: absorbing, reward 0
assert np.allclose(P.sum(-1), 1)                        # every row is a probability distribution

def evaluate(P, R, pi, gamma):          # P: (S, A, S), R: (S, A), pi: (S, A)
    P_pi = np.einsum('sa,sat->st', pi, P)
    r_pi = (pi * R).sum(1)
    return np.linalg.solve(np.eye(len(r_pi)) - gamma * P_pi, r_pi)

right = np.zeros((S, A)); right[:, 1] = 1
V = evaluate(P, R, right, 0.9)
print(V.round(2) + 0.0)                 # [0.   4.28 5.66 6.8  8.02 9.44 0.  ]
Q = R + 0.9 * P @ V                     # Q^pi(s, a) from V^pi
print(Q[1].round(2))                    # [1.82 4.28]  left vs right in cell 1

def returns(rewards, dones, gamma):     # walk backwards, reset at episode ends
    G, out = 0.0, np.zeros(len(rewards))
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G * (1 - dones[t])
        out[t] = G
    return out
print(returns([1, 0, 2, 5, 5], [0, 0, 1, 0, 1], 0.9).round(2))  # [2.62 1.8  2.   9.5  5.  ]
```

Terminal states are **absorbing** (they loop to themselves with reward 0), so their value is forced to 0. `Q = R + 0.9 * P @ V` is the one-step look-ahead from $V^\pi$ to $Q^\pi$: in cell 1, right (4.28) beats left (1.82). And `1 - dones[t]` stops the second episode's rewards leaking into the first.

## Checking your MDP and your values

Errors here are silent: the code prints plausible numbers. Cheap checks catch most of them.

- **Probabilities:** `P.sum(-1)` must be all ones, and no entry negative.
- **Terminals:** their values must come out exactly 0 (unless your convention gives a reward on leaving them).
- **Bounds:** every value must lie between $r_{\min}/(1-\gamma)$ and $r_{\max}/(1-\gamma)$. In the corridor that is 0 to 100; a value of 250 means a bug.
- **Bellman residual:** plug the computed $V$ back in. `np.abs(r_pi + gamma * P_pi @ V - V).max()` should be at rounding level (around 1e-12 or smaller).
- **Simulation:** for one or two states, average a few thousand sampled returns and compare, as in the Monte Carlo figure. A mismatch means the model and the simulator disagree.

## What goes wrong

The most common bug is treating a time limit as a real ending. Take a task where the agent earns 1 every step and never fails, with $\gamma=0.9$, so the true value is $1/(1-0.9)=10$. Episodes are cut after 20 steps. If the learner treats step 20 as terminal, the target there is just $r$:

~~~dg-fig 08.01-truncation.svg
Left: TD learning on the same state, treating the 20-step cutoff as terminal (orange) versus bootstrapping through it (green). The bug settles near 6.8 instead of 10. Right: the bug implicitly claims the return depends on the time step, but the state contains no clock.
~~~

One update in twenty is pulled towards 1 instead of $1+0.9V$, so the value settles near $1/(1-0.9\cdot 19/20)\approx 6.9$ (our run ends at 6.82, saw-toothing around it). With no clock in the state, the agent cannot tell "early" from "late".

| Mistake | Symptom | Fix
| --- | --- | --- |
| Time limit treated as terminal | values biased low; behaviour degrades near the cutoff | separate `terminated` and `truncated`; bootstrap on truncation
| Return not reset at `done` | values leak between episodes, terminal states look valuable | multiply by `1 - done` in the backward recursion
| Off-by-one reward index | values shifted by one step; first or last reward lost | fix the convention: $r_{t+1}$ follows $(s_t,a_t)$
| State is not Markov | identical states get conflicting targets; learning plateaus | add history (frame stacks, velocity, remaining time) to the state
| $\gamma$ too low | agent grabs small nearby rewards, ignores distant goals | raise $\gamma$; check the horizon $1/(1-\gamma)$ covers the task
| $\gamma = 1$ in a continuing task | values diverge, the linear system is singular | use $\gamma<1$, or the average-reward formulation
| Rows of $P$ not summing to 1 | values out of the possible range | assert `P.sum(-1) == 1` when you build the model

## Going deeper

**Why the inverse exists.** $P_\pi$ is a stochastic matrix, so its eigenvalues have magnitude at most 1 and those of $\gamma P_\pi$ at most $\gamma<1$; hence $I-\gamma P_\pi$ is invertible and $(I-\gamma P_\pi)^{-1}=\sum_{k\ge 0}\gamma^k P_\pi^k$, which is the definition of the return in matrix form. With $\gamma=1$ the series can diverge, unless every policy reaches a terminal state with probability 1 (a "proper" policy in the stochastic shortest-path setting).

**Why a max equation has a unique solution.** The Bellman optimality operator $(TQ)(s,a)=\sum_{s'}P(s'\mid s,a)[R+\gamma\max_{a'}Q(s',a')]$ is a $\gamma$-**contraction** in the max norm: it shrinks the largest gap between any two $Q$ tables by a factor $\gamma$. By the Banach fixed-point theorem it has exactly one fixed point, $Q^*$, and repeated application converges to it from anywhere. That is value iteration, the subject of 08.02.

**Why a deterministic optimal policy exists.** Given $Q^*$, the greedy policy achieves $\max_a Q^*(s,a)=V^*(s)$ in every state simultaneously. Randomising can only average in worse actions. Stochastic policies matter only when the agent cannot see the full state (a partially observable MDP, or POMDP) or faces an adversary.

**Reward shaping.** Adding extra rewards to speed up learning can change the optimal policy. Adding $\gamma\Phi(s')-\Phi(s)$ for any function $\Phi$ of the state (potential-based shaping, Ng, Harada and Russell, 1999) provably leaves the optimal policy unchanged, because the extra terms telescope along any trajectory.

## Key terms

| Term | Meaning
| --- | --- |
| MDP | States, actions, transition probabilities, rewards and a discount: $(\mathcal S,\mathcal A,P,R,\gamma)$
| Markov property | The next state and reward depend only on the current state and action
| Policy $\pi(a\mid s)$ | The agent's rule: a probability for each action in each state
| Return $G_t$ | Discounted sum of future rewards, $r_{t+1}+\gamma G_{t+1}$
| Discount $\gamma$ | Weight per step on future reward; effective horizon about $1/(1-\gamma)$
| Value $V^\pi(s)$ | Expected return from $s$ when following $\pi$
| Action value $Q^\pi(s,a)$ | Expected return after taking $a$ in $s$, then following $\pi$
| Bellman expectation equation | Linear equation linking each value to its successors' values under $\pi$
| Bellman optimality equation | The same with a max over next actions; its solution gives $Q^*$
| Terminal vs truncated | A real end (no bootstrap) vs an artificial cutoff (bootstrap)

<div class="take">
<div class="lab">Key takeaway</div>

an MDP is states, actions, Markov transitions, rewards and a discount; values are expected discounted returns, the Bellman expectation equation is linear and solvable exactly for a fixed policy, and the Bellman optimality equation adds a max whose greedy solution is the optimal policy.

</div>
