---
id: "08.03"
course: "deep-learning"
module: "08"
slug: "monte-carlo-and-temporal-difference-learning"
title: "Monte Carlo and temporal-difference learning"
keyIdeas: ["Without a model, estimate $V^\\pi$ from experience by nudging $V(s)$ towards a sampled **target**. Monte Carlo uses the observed return $G_t$: unbiased, but high-variance, and it must wait for the episode to end.", "TD(0) moves $V(s)$ towards $r+\\gamma V(s')$, reusing its own estimate of the next state (**bootstrapping**). The target is biased while $V$ is wrong, but far less noisy, and TD learns online after every step.", "The **TD error** $\\delta_t=r_{t+1}+\\gamma V(s_{t+1})-V(s_t)$ says how much better or worse a step went than predicted. It drives TD, SARSA, Q-learning, actor-critic and GAE.", "**n-step returns** and the **$\\lambda$-return** interpolate between TD(0) ($n=1$, $\\lambda=0$) and Monte Carlo ($n=\\infty$, $\\lambda=1$); intermediate values usually learn fastest, and the best step size shrinks as $n$ or $\\lambda$ grows.", "**Eligibility traces** implement the $\\lambda$-return online by spreading each TD error back over recently visited states with weight $(\\gamma\\lambda)^k$, which makes sparse, delayed rewards propagate much faster than TD(0)."]
question: "In a 1,000-step episodic task where the only reward arrives at the end, you compare Monte Carlo with TD(0) for value prediction, each run for the same small number of episodes. Which learns faster for states near the start, and why? How would TD($\\lambda$) with $\\lambda=0.9$ change the picture?"
answer: "TD(0) propagates the final reward back by only one state per visit. After the first episode only the second-to-last state has learned anything (the corridor figure shows exactly this: after one episode only the final state has moved), and the information reaches the start only after on the order of 1,000 episodes along the path (sooner if trajectories overlap and $\\alpha$ is large). Monte Carlo updates every visited state with the actual return after each episode, so states near the start get a useful, if noisy, signal immediately. With a sparse, delayed reward and few episodes, MC therefore learns the early states faster; TD's lower variance only pays off once values have propagated. TD($\\lambda$) with $\\lambda=0.9$ spreads each TD error back over recently visited states with weight $(\\gamma\\lambda)^k$, so the final reward reaches tens of steps back in one episode: the trace falls to 1 percent after about 44 steps for $\\gamma\\lambda=0.9$. That is much faster than TD(0) while keeping lower variance than full MC, but on a 1,000-step chain it still covers only the last few dozen states per episode. For very long chains, $\\lambda$ closer to 1 or long n-step returns help further; with accumulating traces, reduce $\\alpha$ as $\\lambda$ approaches 1, since large $\\lambda$ with a large step size can make the values diverge."
sources: [{"title": "Sutton and Barto: Reinforcement Learning, An Introduction (2nd ed.)", "url": "https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf", "note": "Chapters 5, 6, 7 and 12: Monte Carlo, TD(0), n-step methods and eligibility traces; the random-walk comparisons."}, {"title": "Sutton: Learning to Predict by the Methods of Temporal Differences", "url": "https://link.springer.com/article/10.1007/BF00992698", "note": "The 1988 paper that introduced TD learning and TD(lambda)."}, {"title": "David Silver: UCL Course on RL", "url": "https://www.davidsilver.uk/teaching/", "note": "Lecture 4 (model-free prediction): the bias-variance picture of MC vs TD."}]
practice: ["td-0-value-update", "monte-carlo-policy-evaluation", "discounted-returns"]
tags: ["TD learning", "Monte Carlo", "eligibility traces"]
tools: ["NumPy"]
---

## What you will learn

- How to estimate a policy's value $V^\pi$ from experience alone, with no model.
- **Monte Carlo** prediction: average the returns you saw. Unbiased, noisy, episode-bound.
- **Temporal-difference** learning, TD(0): learn after every step by bootstrapping, driven by the **TD error**.
- How to run both by hand on a tiny random walk, then in NumPy, and read the learning curves.
- **n-step returns**, the **$\lambda$-return** and **eligibility traces**, which sit between the two extremes.

*Before you start:* returns, discounting and $V^\pi$ from 08.01 (Markov decision processes), and the Bellman expectation backup from 08.02 (dynamic programming).

Lesson 08.02 computed $V^\pi$ by sweeping Bellman backups, which needs the transition probabilities $P(s'\mid s,a)$. Real problems rarely hand you those. This lesson keeps the same goal, evaluating a fixed policy (the **prediction** problem), but learns from **samples**: run the policy, watch what happens, nudge the estimates. Lesson 08.04 turns these into control methods.

## The idea in one picture

The standard test bed is a five-state **random walk**: states A to E in a row, every episode starting at C. Each step moves left or right with equal probability. Falling off the left end gives reward 0, off the right end reward 1. With no discount ($\gamma = 1$), a state's value is the probability of finishing on the right.

| State | A | B | C | D | E
| --- | --- | --- | --- | --- | --- |
| True value $V^\pi(s)$ | 1/6 ≈ 0.17 | 2/6 ≈ 0.33 | 3/6 = 0.50 | 4/6 ≈ 0.67 | 5/6 ≈ 0.83

Pretend we do not know this table: start every estimate at 0.5 and let the agent wander.

~~~dg-fig 08.03-walk.svg
TD(0) estimates on the random walk (step size 0.1, one run). After episode 1 only A has moved (that episode ended on the left); after 100 the line is close to the true values (dashed).
~~~

Nothing about the environment was ever written down; the estimates moved only because the agent experienced transitions. Each update was local: "the state I just left is worth roughly what I got plus what I think the next state is worth". That sentence is temporal-difference learning.

## Step 1: one update rule, different targets

Every method in this lesson uses the same template, a running average:

$$V(s_t) \leftarrow V(s_t) + \alpha\,\big[\text{target} - V(s_t)\big].$$

- $V(s_t)$ is the current estimate for the state visited at time $t$.
- The **target** is a sampled guess of what $V(s_t)$ should be.
- $\alpha$ (between 0 and 1) is the **step size**: how far to move towards the target. With $\alpha = 0.1$ you close 10% of the gap.

Number check: $V = 0.5$, target $= 1$, $\alpha = 0.1$ gives $0.5 + 0.1 \times 0.5 = 0.55$.

If the target is right on average, repeated updates settle at the right answer. With a constant $\alpha$ the estimate keeps jiggling around it; a slowly shrinking $\alpha$ makes it converge exactly. The methods below differ only in the target.

## Step 2: Monte Carlo uses the real return

The most obvious target is what actually happened from that visit onward: the **return**

$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots + \gamma^{T-t-1} r_T,$$

where $r_{t+1}$ is the reward after leaving $s_t$, $\gamma$ the discount and $T$ the end time. By 08.01, $V^\pi(s) = \mathbb E[G_t \mid s_t = s]$, so $G_t$ is an **unbiased** target: on average it is exactly right.

Monte Carlo (MC) prediction waits for the episode to finish, walks backwards computing each $G_t$, and moves every visited state towards its return. **First-visit MC** uses only the first visit to $s$ per episode; **every-visit MC** uses all of them. Both converge to $V^\pi$.

- **No bias**: it averages real outcomes and never relies on other estimates.
- **No Markov assumption**: it just averages what followed, whatever the state description.
- **High variance**: $G_t$ adds up every random reward and transition until the end, so long episodes make it very noisy.
- **Episode-bound**: no update until the episode ends; continuing tasks need truncation.

## Step 3: TD(0) bootstraps from the next state

Temporal-difference learning does not wait. After one step it has a real reward $r_{t+1}$ and an estimate for the state $s_{t+1}$ it landed in, which stands in for everything after:

$$V(s_t) \leftarrow V(s_t) + \alpha\,\delta_t, \qquad \delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t).$$

- $r_{t+1} + \gamma V(s_{t+1})$ is the **TD target**: one real reward, then a guess.
- $\delta_t$ is the **TD error**: how much better ($\delta > 0$) or worse ($\delta < 0$) this step went than $V(s_t)$ predicted.
- A terminal state has value 0 by definition, so at the last step the target is just the final reward.

Using your own estimate inside the target is **bootstrapping**: the Bellman backup of 08.02 with the sum over next states replaced by the one you reached.

Number check: $V(s) = 0.5$, reward $r = 1$, next state $V(s') = 0.6$, $\gamma = 0.9$, $\alpha = 0.1$. Then $\delta = 1 + 0.9 \times 0.6 - 0.5 = 1.04$ and $V(s) \leftarrow 0.5 + 0.1 \times 1.04 = 0.604$. MC would wait for the full return, say $G = 2.3$, giving $0.5 + 0.1 \times 1.8 = 0.68$; but that return could just as easily have been 0.

<div class="ex">
<div class="lab">Worked example</div>

**two episodes of the random walk by hand**

Start with $V = 0.5$ for A to E, $\gamma = 1$, $\alpha = 0.1$. Episode 1 goes C → D → E → off the right end (reward 1).

TD(0). Leaving C: $\delta = 0 + V(D) - V(C) = 0$. Leaving D: $\delta = 0 + 0.5 - 0.5 = 0$. Leaving E: $\delta = 1 + 0 - 0.5 = 0.5$, so $V(E) = 0.5 + 0.1 \times 0.5 = 0.55$. Only E learned anything.

Monte Carlo: the return from C, D and E is $G = 1$, so each moves to $0.5 + 0.1 \times (1 - 0.5) = 0.55$. All three learned at once.

Episode 2 goes C → B → C → D → E → off the right end. TD(0): the first three steps give $\delta = 0$. Leaving D: $\delta = 0 + V(E) - V(D) = 0.55 - 0.5 = 0.05$, so $V(D) = 0.505$. Leaving E: $\delta = 1 + 0 - 0.55 = 0.45$, so $V(E) = 0.55 + 0.045 = 0.595$.

The pattern: in TD(0) the reward travels back one state per visit. MC pushes it to every visited state at once, but each push is only as reliable as one random return.

</div>

## Step 4: bias versus variance

Why prefer a target that leans on a possibly wrong estimate? Noise. Take a chain of 10 states traversed left to right, where every step pays a random reward with mean $-1$ and standard deviation 1. The true value of the first state is $-10$.

~~~dg-fig 08.03-targets.svg
20,000 sampled targets for the first state. The MC return (orange) is centred on −10 but wide; the TD target with a correct V(s′) (blue) is centred there too and much narrower. With a wrong V(s′) (yellow) it is just as narrow but centred on −6: bootstrapping bias.
~~~

The MC return adds ten noisy rewards, so its variance is about 10 (measured: 9.93). The TD target contains one noisy reward plus a fixed number, so its variance is about 1 (measured: 0.99). Ten times less noise means the average settles faster. The price is the yellow histogram: while $V(s')$ is wrong, the TD target is consistently off. That bias shrinks as TD fixes $V(s')$ too.

The **Markov property** makes this work: once in $s'$, how you got there no longer matters, so every path through $s'$ shares one estimate. MC cannot share like that.

## Step 5: compare them on the random walk

Which reaches a good estimate in fewer episodes? Here is the root-mean-square (RMS) error over the five states, averaged over 100 runs:

~~~dg-fig 08.03-rms.svg
RMS error against episodes, averaged over 100 runs. Every TD curve (solid) beats every MC curve (dashed). Larger step sizes fall faster but level off higher.
~~~

First, TD is ahead at every step size tried: after 100 episodes TD with $\alpha = 0.05$ is at 0.036, the best MC curve around 0.09. Second, step size matters as much as method: $\alpha = 0.15$ reaches 0.06 in about 20 episodes but goes no lower, while $\alpha = 0.05$ starts slower and ends better. Decaying $\alpha$ gets both.

## Monte Carlo and TD in code

The whole of both methods fits in a few lines of NumPy. Terminal states are stored with value 0 and never updated.

```python
import numpy as np

def episode(rng):
    # 5-state random walk: states 1..5 are A..E, 0 and 6 are terminal; reward 1 only on exiting right
    s, traj = 3, []
    while s not in (0, 6):
        s2 = s + rng.choice([-1, 1])
        traj.append((s, 1.0 if s2 == 6 else 0.0, s2))
        s = s2
    return traj

def td0(V, traj, alpha, gamma=1.0):
    for s, r, s2 in traj:                       # update after every step
        V[s] += alpha * (r + gamma * V[s2] - V[s])

def mc(V, traj, alpha, gamma=1.0):
    G = 0.0
    for s, r, s2 in reversed(traj):             # walk backwards to build returns
        G = r + gamma * G
        V[s] += alpha * (G - V[s])              # every-visit, constant step size

true_V = np.arange(1, 6) / 6
for name, update, alpha in [("TD(0)", td0, 0.05), ("MC", mc, 0.03)]:
    errs = []
    for seed in range(50):                      # one run is noisy: average 50
        rng = np.random.default_rng(seed)
        V = np.full(7, 0.5); V[[0, 6]] = 0.0    # terminal values must stay 0
        for _ in range(100):
            update(V, episode(rng), alpha)
        errs.append(np.sqrt(np.mean((V[1:6] - true_V) ** 2)))
    print(name, "last run:", np.round(V[1:6], 2), "mean RMS:", round(np.mean(errs), 3))
# TD(0) last run: [0.16 0.35 0.52 0.71 0.88] mean RMS: 0.038
# MC last run: [0.22 0.48 0.65 0.71 0.86] mean RMS: 0.096
```

`td0` could run inside the environment loop, one transition at a time; `mc` needs the finished trajectory.

## Step 6: n-step returns sit in between

MC uses all the rewards; TD(0) uses one. Use $n$ real rewards and then bootstrap, and you get the **n-step return**:

$$G_t^{(n)} = r_{t+1} + \gamma r_{t+2} + \dots + \gamma^{n-1} r_{t+n} + \gamma^n V(s_{t+n}).$$

With $n = 1$ it is the TD target. If the episode ends before $t + n$, the bootstrap term is dropped, so $n = \infty$ is Monte Carlo.

~~~dg-layers Targets from least to most bootstrapping; bias falls and variance rises as you go up.
Monte Carlo | target $G_t$: all real rewards, no bootstrap
n-step | target $r_{t+1}+\dots+\gamma^{n-1}r_{t+n}+\gamma^nV(s_{t+n})$
*TD(0) | target $r_{t+1}+\gamma V(s_{t+1})$: one reward, then bootstrap
DP | target is the full expectation over $s'$ from the model
~~~

Each update carries reward back $n$ states instead of one, while the target holds only $n$ noisy rewards. Intermediate $n$, often 3 to 10, usually beats both ends; Rainbow DQN, for example, uses 3-step targets.

~~~dg-fig 08.03-nstep.svg
19-state random walk (reward −1 left, +1 right), error over the first 10 episodes, 50 runs. Left: n = 4 is best (0.27); one-step TD (0.35) and Monte Carlo (0.48) are worse. Right: TD(λ) shows the same U-shape, λ = 0.8 best (0.27). Curves stop where error left the chart.
~~~

The best step size depends on $n$: longer returns are noisier and want a smaller $\alpha$ (0.8 for $n = 1$, 0.4 for $n = 4$, 0.1 for $n = 16$). Tune them together.

## Step 7: the lambda-return averages every n

Rather than pick one $n$, the **$\lambda$-return** blends all of them with geometrically shrinking weights:

$$G_t^\lambda = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}.$$

- $\lambda \in [0, 1]$ controls how far you look ahead before trusting estimates. The weights add up to 1; in an episodic task the weight of returns longer than the episode lands on the full return.
- $\lambda = 0$ puts all weight on $n = 1$: TD(0). $\lambda = 1$ puts all weight on the full return: Monte Carlo.

Number check: for $\lambda = 0.5$ the weights are 0.5, 0.25, 0.125, ...; for $\lambda = 0.9$ they are 0.1, 0.09, 0.081, ... and fade slowly. The same weighting reappears in 08.06 as Generalised Advantage Estimation (GAE).

## Step 8: eligibility traces do it online

The $\lambda$-return needs future rewards, so taken literally you wait for the episode to end. **Eligibility traces** get the same effect looking backwards. Each state keeps a trace $e(s)$: how much credit it deserves for what happens now.

- every step, all traces fade: $e(s) \leftarrow \gamma\lambda\, e(s)$;
- the state just left gets bumped: $e(s_t) \leftarrow e(s_t) + 1$;
- the current TD error updates every state in proportion to its trace: $V(s) \leftarrow V(s) + \alpha\,\delta_t\, e(s)$.

~~~dg-fig 08.03-trace.svg
Left: trace of one state visited at steps 2, 5, 6 and 12, γλ = 0.8. An accumulating trace adds 1 per visit (peak 2.21); a replacing trace resets to 1. Right: λ-return weights on each n-step return.
~~~

A TD error credits the state just left at full weight, the one before at $\gamma\lambda$, then $(\gamma\lambda)^2$, and so on. With $\gamma\lambda = 0.9$ the credit falls below 1% after about 44 steps ($0.9^{44} \approx 0.01$). This is called the **backward view**; the $\lambda$-return is the **forward view**.

```python
def td_lambda(V, traj, alpha, gamma=1.0, lam=0.8):
    e = np.zeros_like(V)                        # one trace per state, reset each episode
    for s, r, s2 in traj:
        delta = r + gamma * V[s2] - V[s]
        e *= gamma * lam                        # every trace fades
        e[s] += 1.0                             # accumulating trace for the state just left
        V += alpha * delta * e                  # credit all recent states at once
        V[[0, 6]] = 0.0
# Same 5-state walk, alpha = 0.05, 100 episodes, 50 runs:
# TD(0.8) mean RMS: 0.065
```

On that tiny walk, at TD(0)'s step size, $\lambda = 0.8$ lands between TD(0) (0.038) and MC (0.096). Traces earn their keep when rewards are far away:

~~~dg-fig 08.03-sparse.svg
Values after one episode down a 40-state corridor (step size 0.5, reward 1 at the end). MC moves every state to 0.5, TD(0) only the last. TD(0.9) credits the path with weight shrinking by 0.9 per step back: 0.45 for state 39, 0.008 for state 1.
~~~

## Reading the result

Usually you do not know the true values. Instead:

- **Watch the TD errors.** For a fixed policy their running mean should drift to 0 and their spread shrink. A mean stuck away from 0 means values are systematically off.
- **Check against Monte Carlo on fresh episodes.** Average real returns from a few start states and compare with $V$: slow, but unbiased.
- **Try a few step sizes.** A curve that plateaus high means $\alpha$ is too large; one that barely moves, too small.
- **Look along a typical path.** In sparse-reward tasks, values non-zero only near the goal mean credit is not propagating; raise $\lambda$ or $n$.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Values never settle | Constant $\alpha$ too large | Smaller or decaying $\alpha$
| Early states stay at their initial guess | TD(0) with sparse, delayed reward | n-step returns or larger $\lambda$
| Values blow up to inf | Accumulating traces, large $\lambda$ and $\alpha$ (19-state walk, $\lambda = 1$, $\alpha = 0.2$: 29 of 50 runs diverged) | Smaller $\alpha$, replacing traces, true online TD($\lambda$)
| Terminal state gets a non-zero value | Updating the terminal state | Store terminal values as 0, never update
| TD worse than MC on a partially observed task | State is not Markov, so bootstrapping shares values wrongly | Add history to the state, or larger $\lambda$

A summary of the trade-offs:

| Method | Bias | Variance | Updates | Needs episode end
| --- | --- | --- | --- | --- |
| Monte Carlo | none | high | end of episode | yes
| TD(0) | yes, from bootstrapping | low | every step | no
| n-step TD | less | medium | after n steps | no
| TD($\lambda$) with traces | tunable | tunable | every step | no

## Going deeper

**Batch updating shows what each method estimates.** Replay a fixed set of episodes until values stop changing. Batch MC converges to the values minimising squared error against observed returns. Batch TD(0) converges to the value function of the **maximum-likelihood Markov model**: the MDP whose transition probabilities are the observed frequencies (the **certainty-equivalence** estimate). Sutton and Barto's example: eight episodes; one goes A → B → end with rewards 0, 0; six start in B and end with reward 1; one starts in B and ends with 0. Both methods give $V(B) = 6/8 = 0.75$. MC gives $V(A) = 0$, the only return ever seen from A. TD gives $V(A) = 0.75$, because A always led to B. If the process is Markov, TD's answer generalises better.

**Convergence.** Tabular TD(0) converges to $V^\pi$ with probability 1 if $\sum \alpha_t = \infty$ and $\sum \alpha_t^2 < \infty$. With linear function approximation and on-policy data, TD($\lambda$) still converges, to a point whose error is within a factor $(1 - \gamma\lambda)/(1 - \gamma)$ of the best achievable. Off-policy data plus function approximation plus bootstrapping (the "deadly triad") can diverge.

**Forward and backward views.** With offline updates (summed and applied at episode end), accumulating-trace TD($\lambda$) gives exactly the offline $\lambda$-return update; with $\lambda = 1$ that is every-visit MC. Online they differ slightly; **true online TD($\lambda$)** (van Seijen and Sutton, 2014) adds a correction that makes them match exactly for linear function approximation.

**Traces with function approximation.** For $\hat v(s; w)$ the trace is a vector over parameters: $z \leftarrow \gamma\lambda z + \nabla_w \hat v(s_t; w)$, $w \leftarrow w + \alpha\,\delta_t z$. Deep RL usually computes the forward view over stored segments instead (n-step targets, GAE), because traces fit poorly with replay.

**Why the TD error matters.** $\delta_t$ reappears in SARSA and Q-learning (08.04), as the advantage signal in actor-critic (08.06), and in neuroscience as a model of dopamine responses.

## Key terms

| Term | Meaning
| --- | --- |
| Prediction | Estimating $V^\pi$ for a fixed policy
| Return $G_t$ | Discounted sum of rewards actually received from $t$ to the episode's end
| Monte Carlo prediction | Moving $V(s)$ towards observed returns
| Bootstrapping | Building a target partly from your own current estimates
| TD error $\delta_t$ | $r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$: how much better or worse a step went than predicted
| n-step return | $n$ real rewards followed by a bootstrapped estimate of the rest
| $\lambda$-return | Geometrically weighted average of all n-step returns; $\lambda = 0$ is TD(0), $\lambda = 1$ is MC
| Eligibility trace | Per-state memory of recent visits, decaying by $\gamma\lambda$, that spreads each TD error backwards

<div class="take">
<div class="lab">Key takeaway</div>

model-free prediction moves values towards a sampled target. Monte Carlo uses the full observed return (unbiased, noisy, episode-bound), TD(0) uses one reward plus the next state's estimate (biased while values are wrong, low-variance, online), and n-step returns, $\lambda$-returns and eligibility traces span everything in between. The TD error $\delta_t$ is the signal that drives the rest of reinforcement learning.

</div>
