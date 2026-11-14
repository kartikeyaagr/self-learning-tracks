---
id: "08.04"
course: "deep-learning"
module: "08"
slug: "q-learning-and-sarsa"
title: "Q-learning and SARSA"
keyIdeas: ["Control without a model needs action values: with $Q(s,a)$ you can act greedily without knowing $P$.", "SARSA is on-policy: its target $r+\\gamma Q(s',a')$ uses the action the behaviour policy actually takes next, so it learns the value of the exploring policy.", "Q-learning is off-policy: its target $r+\\gamma\\max_{a'}Q(s',a')$ learns $Q^*$ whatever policy generated the data, which is what makes replay buffers possible.", "Expected SARSA averages over the next action under the policy, removing the sampling noise of $a'$; with a greedy target policy it becomes Q-learning.", "The max in Q-learning overestimates noisy values (maximisation bias); Double Q-learning separates selection from evaluation. DQN, covered in AI Research, is Q-learning with a network, replay and a target network."]
question: "You train a tabular Q-learning agent from a fixed log of transitions collected by a cautious human operator who never visits states near a hazard. The learned greedy policy confidently routes through the hazard region and fails. Q-learning is off-policy and 'learns $Q^*$ from any data', so what went wrong?"
answer: "Off-policy convergence to $Q^*$ requires that the behaviour data keep visiting every relevant state-action pair. The operator never visited the hazard region, so $Q$ there stays at its initial values and is never corrected by real rewards. If initial values are zero or optimistic and rewards elsewhere are negative per step, untried actions near the hazard look as good as or better than tried ones, and the max in the Q-learning target propagates those unearned values back to states that were visited. The greedy policy then exploits estimation error rather than real knowledge. This is the core failure of offline RL: distribution shift between the data and the learned policy. Remedies include collecting data that covers the region, initialising unvisited values pessimistically, or constraining the policy to actions well supported by the data (conservative or behaviour-regularised methods)."
sources: [{"title": "Sutton and Barto: Reinforcement Learning, An Introduction (2nd ed.)", "url": "https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf", "note": "Chapter 6.4 to 6.7: SARSA, Q-learning, Expected SARSA, cliff walking and Double Q-learning."}, {"title": "Mnih et al.: Playing Atari with Deep Reinforcement Learning", "url": "https://arxiv.org/abs/1312.5602", "note": "Q-learning with a neural network and experience replay; the step from this lesson to deep RL."}, {"title": "David Silver: UCL Course on RL", "url": "https://www.davidsilver.uk/teaching/", "note": "Lecture 5 (model-free control): GLIE, SARSA and Q-learning convergence."}]
practice: ["q-learning-update", "sarsa-update", "expected-sarsa", "epsilon-greedy-action-selection"]
tags: ["Q-learning", "SARSA", "off-policy"]
tools: ["NumPy"]
---

## From prediction to control

Lesson 08.03 estimated $V^\pi$ for a fixed policy. **Control** means finding a good policy. Knowing $V$ is not enough without a model, because choosing an action requires looking one step ahead through $P$. Learning $Q(s,a)$ instead removes the need: the greedy action is $\arg\max_aQ(s,a)$. The recipe is generalised policy iteration (08.02) with samples: update $Q$ with TD, and keep the policy (mostly) greedy with respect to it.

That raises the exploration problem straight away. A purely greedy agent never tries actions it currently rates poorly, so it never learns they were better. The standard fix is **$\varepsilon$-greedy**: with probability $\varepsilon$ take a uniformly random action, otherwise the greedy one. Lesson 08.05 treats exploration properly; here $\varepsilon$ is usually decayed (say from 1.0 to 0.05) over training.

## SARSA: on-policy TD control

SARSA is named after the tuple it uses, $(s,a,r,s',a')$:

$$Q(s,a)\leftarrow Q(s,a)+\alpha\big[r+\gamma Q(s',a')-Q(s,a)\big]$$

where $a'$ is the action the agent will actually take in $s'$, chosen by the same $\varepsilon$-greedy policy. SARSA therefore evaluates and improves the policy it is executing, exploration included. It converges to $Q^*$ if exploration decays appropriately (greedy in the limit with infinite exploration, "GLIE"); with fixed $\varepsilon$ it converges to the best $\varepsilon$-greedy policy.

## Q-learning: off-policy TD control

Q-learning (Watkins, 1989) replaces the next action's value with the best next action's value:

$$Q(s,a)\leftarrow Q(s,a)+\alpha\big[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\big]$$

The target no longer depends on what the agent does next, so the data can come from any behaviour policy that keeps visiting all state-action pairs: an $\varepsilon$-greedy version of itself, an old policy, a human, a replay buffer. Q-learning is a sampled version of value iteration's optimality backup and converges to $Q^*$ in the tabular case under standard step-size conditions.

~~~dg-flow Same transition, different bootstrap: SARSA follows the next action it will really take; Q-learning assumes the best one.
[SARSA] s, a -(r)-> s prime -(behaviour picks a prime)-> *Q of s prime, a prime
[Q-learning] s, a -(r)-> s prime -(max over actions)-> *max Q of s prime
~~~

| Aspect | SARSA | Expected SARSA | Q-learning
| --- | --- | --- | --- |
| Target | $r+\gamma Q(s',a')$ | $r+\gamma\sum_{a'}\pi(a'\mid s')Q(s',a')$ | $r+\gamma\max_{a'}Q(s',a')$
| Learns | $Q$ of the behaviour policy | $Q$ of the target policy $\pi$ | $Q^*$
| On or off policy | on | either | off
| Target variance | highest (samples $a'$) | lower | low, but biased upwards

## Cliff walking: the difference in behaviour

In the classic cliff-walking gridworld, the shortest path runs along a cliff edge; stepping off costs $-100$ and every other step $-1$. With $\varepsilon=0.1$, Q-learning learns the optimal edge-hugging path, because its target assumes greedy behaviour, but during training its random 10 percent of moves regularly push it off the cliff, so its online reward is worse. SARSA accounts for its own exploration and learns a safer path one row further from the edge, earning more reward while exploring. Neither is wrong: Q-learning answers "what is optimal", SARSA "what is best given that I will keep exploring". For a robot that keeps exploring in the real world, the SARSA answer is often the one you want.

<div class="ex">
<div class="lab">Worked example</div>

**the two targets on one transition**

In $s'$, $Q(s',\cdot)=(2.0, 5.0)$, the reward was $r=1$, $\gamma=0.9$, and the $\varepsilon$-greedy policy happened to sample the exploratory action 1 (value 2.0). Current $Q(s,a)=3.0$, $\alpha=0.5$.

SARSA target: $1+0.9\cdot2.0=2.8$, so $Q(s,a)\to3.0+0.5(2.8-3.0)=2.9$.

Q-learning target: $1+0.9\cdot5.0=5.5$, so $Q(s,a)\to3.0+0.5(5.5-3.0)=4.25$.

Expected SARSA with $\varepsilon=0.1$ over 2 actions ($\pi=(0.05, 0.95)$): target $1+0.9(0.05\cdot2+0.95\cdot5)=5.365$, no sampling noise from $a'$.

</div>

```python
def q_update(Q, s, a, r, s2, done, alpha, gamma):
    target = r if done else r + gamma * Q[s2].max()      # no bootstrap past a true terminal
    Q[s, a] += alpha * (target - Q[s, a])
```

## Maximisation bias and the step to deep RL

$\mathbb E[\max_a\hat Q(s,a)]\ge\max_a\mathbb E[\hat Q(s,a)]$: when estimates are noisy, the max picks whichever is luckily high, so Q-learning's targets are biased upwards, and the bias compounds through bootstrapping. **Double Q-learning** keeps two estimates and uses one to choose the action and the other to evaluate it, $r+\gamma Q_B(s',\arg\max_aQ_A(s',a))$. Optimistic initial values (e.g. $Q_0=+5$ when rewards are at most 1) are a cheap exploration aid in tabular settings: every untried action looks attractive until tried.

Replacing the table with a neural network gives DQN, which adds an experience replay buffer (possible because Q-learning is off-policy) and a slowly updated target network to stabilise the bootstrapped target. The AI Research course covers DQN and its variants (double, dueling, prioritised replay) in depth.

<div class="take">
<div class="lab">Key takeaway</div>

SARSA bootstraps from the next action actually taken and learns the value of the exploring policy; Q-learning bootstraps from the best next action and learns $Q^*$ from any data, at the price of maximisation bias; Expected SARSA sits between them.

</div>
