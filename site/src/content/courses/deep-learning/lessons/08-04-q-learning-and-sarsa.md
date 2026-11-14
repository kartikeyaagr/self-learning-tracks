---
id: "08.04"
course: "deep-learning"
module: "08"
slug: "q-learning-and-sarsa"
title: "Q-learning and SARSA"
keyIdeas: ["Without a model you need **action values**: with $Q(s,a)$ the greedy action is just $\\arg\\max_a Q(s,a)$, no transition probabilities required.", "SARSA is **on-policy**: its target $r+\\gamma Q(s',a')$ uses the next action the $\\varepsilon$-greedy policy really takes, so it learns the value of the exploring policy and avoids danger it might stumble into.", "Q-learning is **off-policy**: its target $r+\\gamma\\max_{a'}Q(s',a')$ learns $Q^*$ whatever policy generated the data, which is what makes replay buffers and learning from logs possible.", "On cliff walking with $\\varepsilon=0.1$, Q-learning learns the optimal edge route but earns less while training (it keeps falling off); SARSA learns a safer detour and earns more.", "Expected SARSA averages over the next action, removing the noise of sampling $a'$; with a greedy target policy it becomes Q-learning.", "The max over noisy estimates is biased upwards (maximisation bias); Double Q-learning uses one table to pick the action and another to score it. DQN is Q-learning with a network, replay and a target network."]
question: "You train a tabular Q-learning agent from a fixed log of transitions collected by a cautious human operator who never visits states near a hazard. The learned greedy policy confidently routes through the hazard region and fails. Q-learning is off-policy and 'learns $Q^*$ from any data', so what went wrong, and what would you change?"
answer: "Off-policy convergence to $Q^*$ has a condition that is easy to forget: the behaviour data must keep visiting every relevant state-action pair. The operator never went near the hazard, so the $Q$ entries there were never updated by a real reward and still hold their initial values. With per-step rewards negative elsewhere (say $-1$) and zero or optimistic initial values, those untouched entries look as good as or better than the honestly estimated ones. The max in the Q-learning target, $r+\\gamma\\max_{a'}Q(s',a')$, then copies the unearned values back into states that were visited and adjacent to the gap, and the greedy policy follows them. It is exploiting estimation error, not knowledge. This is the central failure of offline reinforcement learning: distribution shift between the data and the policy being learned, made worse by maximisation bias. Fixes, roughly in order of effort: (1) check coverage first by counting visits per state-action pair; (2) initialise unvisited entries pessimistically (for example at the worst possible return) so the max ignores them; (3) restrict the greedy choice to actions that appear in the data for that state, or penalise rarely seen actions (the idea behind conservative and behaviour-regularised offline methods); (4) if possible, collect data that covers the hazard region, for example with a safe simulator. Double Q-learning reduces the noise-driven part of the overestimate but does not fix missing coverage, because there is no data at all to correct those entries."
sources: [{"title": "Sutton and Barto: Reinforcement Learning, An Introduction (2nd ed.)", "url": "https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf", "note": "Chapter 6.4 to 6.7: SARSA, Q-learning, Expected SARSA, cliff walking and Double Q-learning."}, {"title": "Mnih et al.: Playing Atari with Deep Reinforcement Learning", "url": "https://arxiv.org/abs/1312.5602", "note": "Q-learning with a neural network and experience replay; the step from this lesson to deep RL."}, {"title": "David Silver: UCL Course on RL", "url": "https://www.davidsilver.uk/teaching/", "note": "Lecture 5 (model-free control): GLIE, SARSA and Q-learning convergence."}]
practice: ["q-learning-update", "sarsa-update", "expected-sarsa", "epsilon-greedy-action-selection"]
tags: ["Q-learning", "SARSA", "off-policy"]
tools: ["NumPy"]
---

## What you will learn

- Why control without a model needs **action values** $Q(s,a)$ rather than state values $V(s)$.
- How $\varepsilon$-greedy exploration works, with the exact probability it gives each action.
- The SARSA, Q-learning and Expected SARSA updates, computed by hand on one transition.
- What **on-policy** and **off-policy** mean, seen in cliff walking, where the two methods learn different paths.
- Why the max in Q-learning overestimates (maximisation bias), how Double Q-learning fixes it, and how all of this leads to DQN.

*Before you start:* MDP vocabulary from 08.01 (return, discount $\gamma$, policy, $Q^*$), the Bellman optimality backup and generalised policy iteration from 08.02, and the TD(0) update from 08.03.

## From predicting values to choosing actions

Lesson 08.03 learned $V^\pi$ for a fixed policy with TD(0). This lesson tackles **control**: find a good policy, still without a model.

The recipe is generalised policy iteration from 08.02 with samples: evaluate a little (one TD step), improve a little (act greedily on the new estimates), repeat. SARSA and Q-learning fill in that recipe differently in exactly one term of the target.

## The idea in one picture

The standard testbed is **cliff walking**: a grid with 4 rows and 12 columns. The agent starts at S in the bottom-left corner and must reach G in the bottom-right. The ten cells between them are a cliff.

| Event | Reward | What happens next
| --- | --- | --- |
| Any normal step (up, right, down, left) | $-1$ | move one cell; walls just block
| Step into a cliff cell | $-100$ | sent back to S, episode continues
| Reach G | $-1$ | episode ends

Every step costs 1, so the agent wants the shortest route. The shortest route runs along the row just above the cliff: 13 steps, return $-13$. We train two agents for 500 episodes each with the same settings (step size $\alpha=0.5$, $\gamma=1$, and 10 percent random moves, $\varepsilon=0.1$), then switch exploration off and draw the route each one would take.

~~~dg-fig 08.04-cliff-paths.svg
Greedy routes after 500 episodes with ε = 0.1. Q-learning hugs the cliff edge (the true shortest path); SARSA takes a detour through the top rows that is safer while it keeps making random moves.
~~~

Same environment, same data-gathering rule, different answers. **Q-learning** found the optimal 13-step edge route. **SARSA** found a 17-step route that keeps two rows between itself and the cliff. Neither has a bug. They are answering different questions:

- Q-learning asks: "if I acted perfectly from now on, what is each action worth?" On the edge row, perfect play never falls, so the edge is best.
- SARSA asks: "given that I will keep taking a random action 10 percent of the time, what is each action worth?" On the edge row, one random "down" costs 100, so the edge is a bad place to be.

This is the difference between **off-policy** learning (learn about a policy other than the one generating the data, here the greedy one) and **on-policy** learning (learn about the policy you are actually running, exploration included).

## Step 1: learn action values, not state values

Suppose you know $V(s)$ for every state. To pick an action you still need to know where each action leads: the greedy choice is $\arg\max_a \sum_{s'} P(s'\mid s,a)\,[R + \gamma V(s')]$, which needs the transition model $P$. Without a model you cannot do that one-step look-ahead.

An **action value** $Q(s,a)$ is the expected return if you take action $a$ in state $s$ and then follow your policy. It already contains the look-ahead, so the greedy action is simply

$$\pi_{\text{greedy}}(s) = \arg\max_a Q(s,a).$$

No $P$ appears, so model-free control learns a table `Q[s, a]`: in the cliff world $48 \times 4 = 192$ entries.

Number check: if in some state $Q(s,\cdot) = (-14, -13, -100, -15)$ for (up, right, down, left), the greedy action is "right", value $-13$. You read it off the table; you never needed to know that "down" lands in the cliff.

## Step 2: explore with $\varepsilon$-greedy

A purely greedy learner never tries actions it currently rates poorly, so it can never discover that one of them was better. The simplest cure is **$\varepsilon$-greedy**: with probability $\varepsilon$ pick an action uniformly at random, otherwise pick the greedy one. With $|A|$ actions, this gives

$$\pi(a\mid s)=\begin{cases}1-\varepsilon+\dfrac{\varepsilon}{|A|} & a=\arg\max_b Q(s,b)\\[4pt] \dfrac{\varepsilon}{|A|} & \text{otherwise.}\end{cases}$$

The greedy action gets the extra $\varepsilon/|A|$ because the random draw can also land on it. Number check in the cliff world ($|A|=4$, $\varepsilon=0.1$): the greedy action has probability $0.9+0.025=0.925$, each other action $0.025$. So on the edge row, the chance of an accidental "down" is 2.5 percent per step, and over the 11 edge steps of the optimal route roughly $1-0.975^{11}\approx 24$ percent of episodes hit the cliff at least once.

In practice $\varepsilon$ is often decayed, say from 1.0 to 0.05. Lesson 08.05 treats exploration properly.

## Step 3: SARSA, the on-policy update

SARSA takes the TD(0) update from 08.03 and applies it to $Q$. After taking action $a$ in state $s$, the agent sees reward $r$ and next state $s'$, then *chooses its next action $a'$ with the same $\varepsilon$-greedy policy*. Only then does it update:

$$Q(s,a)\leftarrow Q(s,a)+\alpha\big[\,r+\gamma\, Q(s',a')-Q(s,a)\,\big].$$

What each symbol is:

- $\alpha$ is the step size (how far to move towards the target, between 0 and 1).
- $r+\gamma Q(s',a')$ is the **TD target**: one real reward plus the estimated value of the action the agent will really take next.
- The bracket is the TD error $\delta$: target minus current estimate.

The name lists what the update uses: $(s, a, r, s', a')$. Because $a'$ comes from the exploring policy, the target includes the occasional random step: sometimes on the edge row $a'$ is "down" and the target includes $-100$.

~~~dg-flow One SARSA step: the next action is chosen first, then used in the target, then actually executed.
State s, action a -(environment)-> reward r, next state s prime -(ε-greedy on Q)-> *next action a prime -(update Q of s, a)-> move to s prime, take a prime
~~~

What it converges to: with a fixed $\varepsilon$, SARSA converges to the best policy *among $\varepsilon$-greedy policies* and its value. If exploration is decayed so that every pair is still tried infinitely often but the policy becomes greedy in the limit (**GLIE**: greedy in the limit with infinite exploration), and the step sizes satisfy the usual conditions, tabular SARSA converges to $Q^*$.

## Step 4: Q-learning, the off-policy update

Q-learning (Watkins, 1989) changes one term. Instead of the value of the next action actually taken, it uses the value of the best next action:

$$Q(s,a)\leftarrow Q(s,a)+\alpha\big[\,r+\gamma\max_{a'}Q(s',a')-Q(s,a)\,\big].$$

The target no longer depends on what the agent does next. It is a sample of the Bellman optimality backup from 08.02, $Q^*(s,a)=\mathbb E[r+\gamma\max_{a'}Q^*(s',a')]$, so Q-learning is value iteration with one sampled transition in place of the expectation over $s'$.

The **behaviour policy** chooses actions and generates data (here $\varepsilon$-greedy); the **target policy** is the one whose value you learn (for Q-learning, greedy). Same policy: on-policy; different: off-policy. Q-learning's data can come from any behaviour policy that keeps visiting every state-action pair: an $\varepsilon$-greedy version of itself, an older policy, a human demonstrator, or transitions replayed from a buffer. In the tabular case, with every pair visited infinitely often and step sizes satisfying $\sum_t\alpha_t=\infty$, $\sum_t\alpha_t^2<\infty$, Q-learning converges to $Q^*$.

~~~dg-flow Same transition, different bootstrap: SARSA follows the next action it will really take; Q-learning assumes the best one.
[SARSA] s, a -(r)-> s prime -(behaviour picks a prime)-> *Q of s prime, a prime
[Q-learning] s, a -(r)-> s prime -(max over actions)-> *max Q of s prime
~~~

## Step 5: Expected SARSA, the average instead of a sample

SARSA's target is noisy because $a'$ is a random draw. Since the agent knows its own policy, it can replace the draw by its average:

$$Q(s,a)\leftarrow Q(s,a)+\alpha\Big[\,r+\gamma\sum_{a'}\pi(a'\mid s')\,Q(s',a')-Q(s,a)\,\Big].$$

This **Expected SARSA** target has the same mean as SARSA's but no noise from $a'$, so it tolerates larger step sizes. With $\pi$ greedy, the sum puts all weight on the best action and it becomes exactly Q-learning.

<div class="ex">
<div class="lab">Worked example</div>

**the three targets on one transition**

Setup: current estimate $Q(s,a)=3.0$, step size $\alpha=0.5$, discount $\gamma=0.9$. The agent received $r=1$ and landed in $s'$, which has two actions with $Q(s',\cdot)=(2.0,\ 5.0)$. Action 1 (value 5.0) is greedy. The $\varepsilon$-greedy policy happened to sample the exploratory action 0 (value 2.0) as $a'$.

SARSA uses the sampled $a'$: target $=1+0.9\times2.0=1+1.8=2.8$. TD error $=2.8-3.0=-0.2$. New value $=3.0+0.5\times(-0.2)=2.9$.

Q-learning uses the best action: target $=1+0.9\times5.0=1+4.5=5.5$. TD error $=5.5-3.0=2.5$. New value $=3.0+0.5\times2.5=4.25$.

Expected SARSA with $\varepsilon=0.1$ over 2 actions: $\pi=(0.05,\ 0.95)$, because the greedy action gets $0.9+0.05$. Expected next value $=0.05\times2.0+0.95\times5.0=0.1+4.75=4.85$. Target $=1+0.9\times4.85=5.365$. New value $=3.0+0.5\times2.365=4.1825$.

Reading: one unlucky exploratory draw dragged SARSA *down* while the others moved *up*. On average SARSA's target equals Expected SARSA's (5.365); Q-learning's is highest because it ignores the 5 percent chance of the worse action.

</div>

| Aspect | SARSA | Expected SARSA | Q-learning
| --- | --- | --- | --- |
| Target | $r+\gamma Q(s',a')$ | $r+\gamma\sum_{a'}\pi(a'\mid s')Q(s',a')$ | $r+\gamma\max_{a'}Q(s',a')$
| Learns | $Q$ of the behaviour policy | $Q$ of the target policy $\pi$ | $Q^*$
| On or off policy | on | either | off
| Target variance | highest (samples $a'$) | lower | low, but biased upwards

## Cliff walking: what the curves show

The route figure showed where each method ends up. The next figure shows what each one *earns while learning*, averaged over 50 independent runs.

~~~dg-fig 08.04-online-return.svg
Reward per episode during training (10-episode moving average, 50 runs). Q-learning knows the best route but keeps falling off it when it explores, so it collects the least reward.
~~~

Over the last 100 episodes Q-learning averaged about $-49$, SARSA $-26$ and Expected SARSA $-21$, though only Q-learning's greedy route is optimal ($-13$): walking the edge, it regularly pays 100 for a random "down".

Across 30 seeds, SARSA's greedy route kept at least one row of buffer in 25 runs, used the edge in 4, and looped in 1. Q-learning chose the 13-step edge every time.

How much exploration matters is the next figure: the same experiment at five values of $\varepsilon$.

~~~dg-fig 08.04-epsilon.svg
Left: as exploration grows, Q-learning's reward while exploring collapses, SARSA's degrades more gently. Right: Q-learning's learned route is always the 13-step edge; SARSA's is a longer detour, and at high ε its greedy route often loops instead of reaching G.
~~~

Reading the numbers (20 runs per point, last 100 episodes): at $\varepsilon=0.01$ the two methods are almost tied (Q-learning $-15.7$, SARSA $-16.5$), because random steps are rare and the edge route is rarely punished. At $\varepsilon=0.1$ SARSA leads, $-26.5$ against $-50.5$. At $\varepsilon=0.3$ the gap is huge: SARSA $-66.8$, Q-learning $-199.8$. Meanwhile Q-learning's greedy route is the 13-step optimum in all 100 runs, whatever $\varepsilon$ was. SARSA's greedy route has a median of 15 to 17 steps, but there is a catch: SARSA's values are only meant to be right for the *exploring* policy, so when you switch exploration off its greedy route is not guaranteed to work. With $\varepsilon=0.3$, 12 of 20 SARSA runs produced a greedy route that loops. If you plan to deploy SARSA's policy greedily, decay $\varepsilon$ during training.

Practical rule: if the agent keeps exploring after deployment, SARSA's answer is the value you will experience; if exploration is only a training device, Q-learning's is the one you want.

## Q-learning and SARSA in code

A complete cliff-walking experiment in plain NumPy. The only line that differs between the two methods is the target.

```python
import numpy as np

ROWS, COLS, START, GOAL = 4, 12, (3, 0), (3, 11)
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]              # up, right, down, left

def step(s, a):
    r = min(max(s[0] + MOVES[a][0], 0), ROWS - 1)
    c = min(max(s[1] + MOVES[a][1], 0), COLS - 1)
    if r == 3 and 1 <= c <= 10:                        # fell off the cliff
        return START, -100.0, False
    return (r, c), -1.0, (r, c) == GOAL

def eps_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(4))                    # explore
    best = np.flatnonzero(Q[s] == Q[s].max())
    return int(rng.choice(best))                       # exploit, random tie-break

def train(method, episodes=500, alpha=0.5, gamma=1.0, eps=0.1, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((ROWS, COLS, 4))
    returns = []
    for _ in range(episodes):
        s, G, done = START, 0.0, False
        a = eps_greedy(Q, s, eps, rng)
        while not done:
            s2, r, done = step(s, a)
            a2 = eps_greedy(Q, s2, eps, rng)           # the action we will really take
            if done:
                target = r                             # no bootstrap past the goal
            elif method == "sarsa":
                target = r + gamma * Q[s2][a2]
            else:                                      # Q-learning
                target = r + gamma * Q[s2].max()
            Q[s][a] += alpha * (target - Q[s][a])
            s, a, G = s2, a2, G + r
        returns.append(G)
    return Q, np.array(returns)

for m in ["sarsa", "q"]:
    Q, R = train(m)
    s, n = START, 0
    while s != GOAL and n < 50:                        # follow the greedy policy
        s, _, _ = step(s, int(Q[s].argmax())); n += 1
    print(f"{m:5s}  online return (last 100 eps) {R[-100:].mean():6.1f}   greedy path {n} steps")
# sarsa  online return (last 100 eps)  -26.2   greedy path 17 steps
# q      online return (last 100 eps)  -61.8   greedy path 13 steps
```

Details that matter:

- **Terminal states**: at the goal the target is just $r$. Bootstrapping from $Q$ of a terminal state adds whatever junk is stored there. If an episode is cut off by a time limit rather than ending for real, you *should* still bootstrap, because the task did not actually end.
- **Order in SARSA**: choose $a'$ before the update and then execute that same $a'$.
- These are one seed; the 50-run average for Q-learning was $-49$, not $-62$.

## Checking that it worked

- **Plot the episode return**, smoothed and averaged over several seeds. It should rise and level off. For a sanity bound, compute the return of a known good policy (here $-13$).
- **Evaluate the greedy policy separately**, with $\varepsilon=0$, every so often. Online return mixes "what was learned" with "the cost of exploring", which is exactly why Q-learning looks worse in the training curve but better at test time.
- **Inspect the table.** $\max_a Q$ at the start should approach $-13$ for Q-learning; values outside the possible range of returns point to a terminal-handling bug.
- **Check coverage.** Count visits per state-action pair. Pairs with zero visits still hold their initial value, and the max in Q-learning will happily trust them.
- **Run several seeds**: with constant step size the route differs between runs.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Values grow without bound or explode | bootstrapping from a terminal state, or $\gamma=1$ in a task that never ends | target $=r$ at true terminals; use $\gamma<1$ for continuing tasks
| Agent repeats one action forever | greedy with no exploration, or argmax always picks index 0 on ties | use $\varepsilon$-greedy; break ties randomly
| Training reward bad but greedy policy fine | Q-learning near a hazard while still exploring | expected; evaluate greedily, or use SARSA if exploration stays on
| Greedy policy loops or keeps changing | constant large $\alpha$, or SARSA run greedily after fixed-ε training | decay $\alpha$ and $\varepsilon$; Expected SARSA
| Confident values for actions never tried | max over uninitialised entries; offline data with gaps | pessimistic initial values, more coverage, conservative offline methods
| SARSA does not reach the optimal path | fixed $\varepsilon$: it is optimal among exploring policies, by design | decay $\varepsilon$ towards 0 (GLIE)

## Going deeper: maximisation bias and Double Q-learning

If every action truly has value 0 but the estimates are noisy, their max is usually above 0: it picks the luckiest estimate. Formally, for any random estimates, $\mathbb E[\max_a\hat Q(s,a)]\ge\max_a\mathbb E[\hat Q(s,a)]$ (Jensen's inequality, since max is convex). For two actions with independent standard normal noise, the max averages about $0.56$; for ten actions about $1.54$ (both checked by simulation, matching the known values $1/\sqrt\pi$ and the expected maximum of 10 normals). Q-learning feeds that inflated max into the target of the previous state, and bootstrapping passes it further back.

The classic demonstration (Sutton and Barto, example 6.7): from state A, "right" ends the episode with reward 0; "left" leads to state B with reward 0, and from B there are ten actions that each end the episode with reward drawn from a normal distribution with mean $-0.1$ and standard deviation 1. Going left is worse on average, so the optimal $\varepsilon$-greedy agent goes left only 5 percent of the time (half of its random moves). But the max over ten noisy estimates in B looks positive early on, so Q-learning is lured left.

~~~dg-fig 08.04-maxbias.svg
Fraction of 1000 runs that choose the bad 'left' action in each episode. Both start near 50 percent (all values tie at zero). Q-learning is then pulled towards 'left' by the max over noisy estimates; Double Q-learning falls quickly towards the 5 percent floor set by random exploration.
~~~

In our run (1000 runs, $\alpha=0.1$, $\varepsilon=0.1$), Q-learning chose "left" in up to 94.8 percent of runs around episode 24, and was still at about 12 percent after 300 episodes. Double Q-learning never rose above its starting coin flip (peak 51 percent, in the first episodes) and ended near 7 percent, close to the 5 percent floor.

**Double Q-learning** removes the bias by never letting one estimate both pick and score an action. It keeps two tables $Q_A$ and $Q_B$. On each step a coin decides which to update; to update $Q_A$:

$$Q_A(s,a)\leftarrow Q_A(s,a)+\alpha\big[\,r+\gamma\,Q_B\big(s',\arg\max_{a'}Q_A(s',a')\big)-Q_A(s,a)\,\big].$$

$Q_A$ picks the action and $Q_B$, with independent noise, scores it, so a lucky value is not confirmed by itself. Behaviour is $\varepsilon$-greedy on $Q_A+Q_B$; Double DQN uses the online and target networks the same way.

A few more points experts care about:

- **Optimistic initial values** (say $Q_0=+5$ when rewards are at most 1) make untried actions look attractive until tried: cheap exploration in tables.
- **Why off-policy is fragile.** Q-learning's off-policy target is what lets DQN use an **experience replay** buffer of old transitions. Combined with function approximation and bootstrapping it forms what Sutton and Barto call the **deadly triad**: any two are fine, all three can diverge. DQN tames it with replay and a slowly updated **target network** that holds the bootstrap target still for thousands of steps. The AI Research course covers DQN and its variants (double, dueling, prioritised replay).
- **Convergence of Q-learning** follows because the optimality backup is a $\gamma$-contraction in the max norm and the sampled update is a stochastic approximation of it; the argument breaks once $Q$ is a neural network.

## Key terms

| Term | Meaning
| --- | --- |
| Control | Finding a good policy, as opposed to evaluating a given one
| Action value $Q(s,a)$ | Expected return after taking $a$ in $s$ and then following the policy
| $\varepsilon$-greedy | Random action with probability $\varepsilon$, greedy action otherwise
| Behaviour / target policy | The policy that generates data / the policy whose value is being learned
| On-policy / off-policy | Behaviour and target policy are the same / different
| SARSA | TD control whose target uses the next action actually taken
| Q-learning | TD control whose target uses the best next action; learns $Q^*$
| Expected SARSA | TD control whose target averages over the next action under $\pi$
| Maximisation bias | Upward bias from taking the max over noisy estimates
| GLIE | Exploration that never stops but becomes greedy in the limit

<div class="take">
<div class="lab">Key takeaway</div>

SARSA bootstraps from the next action it actually takes, so it learns the value of the exploring policy and stays safe while exploring; Q-learning bootstraps from the best next action, so it learns $Q^*$ from any data at the price of risky exploration and maximisation bias; Expected SARSA averages the two ideas, and Double Q-learning removes the bias.

</div>
