---
id: "08.03"
course: "deep-learning"
module: "08"
slug: "monte-carlo-and-temporal-difference-learning"
title: "Monte Carlo and temporal-difference learning"
keyIdeas: ["Without a model, estimate $V^\\pi$ from experience. Monte Carlo moves $V(s)$ towards the observed return $G_t$; it is unbiased but high-variance and must wait for the episode to end.", "TD(0) moves $V(s)$ towards $r+\\gamma V(s')$, using its own estimate of the next state (bootstrapping): biased, lower-variance, and it learns online after every step.", "The TD error $\\delta_t=r_{t+1}+\\gamma V(s_{t+1})-V(s_t)$ is the central quantity of the module: it drives TD, SARSA, Q-learning, actor-critic and GAE.", "n-step returns and the $\\lambda$-return interpolate between TD(0) ($n=1$, $\\lambda=0$) and Monte Carlo ($n=\\infty$, $\\lambda=1$).", "Eligibility traces implement the $\\lambda$-return online by spreading each TD error back over recently visited states with weight $(\\gamma\\lambda)^k$."]
question: "In a 1,000-step episodic task where the only reward arrives at the end, you compare Monte Carlo with TD(0) for value prediction, each run for the same small number of episodes. Which learns faster for states near the start, and why? How would TD($\\lambda$) with $\\lambda=0.9$ change the picture?"
answer: "TD(0) propagates the final reward back by only one state per visit: after the first episode only the second-to-last state has learned anything, and information reaches the start only after on the order of 1,000 episodes along the path (sooner if trajectories overlap and $\\alpha$ is large). Monte Carlo updates every visited state with the actual return after each episode, so states near the start get a useful, if noisy, signal immediately. With a sparse, delayed reward and few episodes, MC learns the early states faster; TD's lower variance only pays off once values have propagated. TD($\\lambda$) with $\\lambda=0.9$ spreads each TD error back over recently visited states with weight $(\\gamma\\lambda)^k$, so the final reward reaches tens of steps back in one episode (the trace falls to 1 percent after about 44 steps for $\\gamma\\lambda=0.9$). It propagates much faster than TD(0) while keeping lower variance than full MC; for very long chains, $\\lambda$ closer to 1 or n-step returns help further."
sources: [{"title": "Sutton and Barto: Reinforcement Learning, An Introduction (2nd ed.)", "url": "https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf", "note": "Chapters 5, 6, 7 and 12: Monte Carlo, TD(0), n-step methods and eligibility traces; the random-walk comparisons."}, {"title": "Sutton: Learning to Predict by the Methods of Temporal Differences", "url": "https://link.springer.com/article/10.1007/BF00992698", "note": "The 1988 paper that introduced TD learning and TD(lambda)."}, {"title": "David Silver: UCL Course on RL", "url": "https://www.davidsilver.uk/teaching/", "note": "Lecture 4 (model-free prediction): the bias-variance picture of MC vs TD."}]
practice: ["td-0-value-update", "monte-carlo-policy-evaluation", "discounted-returns"]
tags: ["TD learning", "Monte Carlo", "eligibility traces"]
tools: ["NumPy"]
---

## From sweeps to samples

Dynamic programming (08.02) computes an expectation over next states using the model. Model-free **prediction** replaces that expectation with samples: run the policy, observe transitions, and nudge value estimates towards targets built from what happened. The general update is

$$V(s_t)\leftarrow V(s_t)+\alpha\,\big[\text{target}-V(s_t)\big]$$

a running average with step size $\alpha$. The methods in this lesson differ only in the target.

## Monte Carlo

The obvious target is the actual return $G_t$ from that visit onward. **Every-visit** MC averages returns after all visits to $s$; **first-visit** MC uses only the first visit per episode (each is a consistent estimator). Properties:

- **Unbiased**: $\mathbb E[G_t\mid s_t=s]=V^\pi(s)$ by definition.
- **High variance**: $G_t$ sums many random rewards and transitions; long episodes make it noisy.
- **Episode-bound**: you cannot update until the episode ends, and it does not work for continuing tasks without truncation.
- **No Markov assumption** needed: it only averages observed returns.

## Temporal-difference learning

TD(0) uses one real reward and the current estimate for the rest:

$$V(s_t)\leftarrow V(s_t)+\alpha\,\delta_t,\qquad \delta_t=r_{t+1}+\gamma V(s_{t+1})-V(s_t)$$

$\delta_t$ is the **TD error**: how much better or worse the step went than predicted. Using $V(s_{t+1})$ inside the target is **bootstrapping**. The target is biased while $V$ is wrong, but it depends on one reward and one transition, so its variance is far lower than MC's. TD updates after every step, works in continuing tasks, and exploits the Markov structure: information propagates through shared successor states. Under standard step-size conditions, tabular TD(0) converges to $V^\pi$.

<div class="ex">
<div class="lab">Worked example</div>

**one TD(0) update**

$V(s)=0.5$, the agent receives $r=1$, lands in $s'$ with $V(s')=0.6$; $\gamma=0.9$, $\alpha=0.1$.

$\delta=1+0.9\cdot0.6-0.5=1.04$, so $V(s)\leftarrow0.5+0.1\cdot1.04=0.604$.

Monte Carlo would instead wait for the whole return, say $G=2.3$, and move $V(s)$ to $0.5+0.1\cdot1.8=0.68$; that single return could just as easily have been 0.

</div>

~~~dg-layers Targets from least to most bootstrapping; bias falls and variance rises as you go up.
Monte Carlo | target $G_t$: all real rewards, no bootstrap
n-step | target $r_{t+1}+\dots+\gamma^{n-1}r_{t+n}+\gamma^nV(s_{t+n})$
*TD(0) | target $r_{t+1}+\gamma V(s_{t+1})$: one reward, then bootstrap
DP | target is the full expectation over $s'$ from the model
~~~

## n-step returns and the lambda-return

Between one step and all steps lie **n-step returns**:

$$G_t^{(n)}=r_{t+1}+\gamma r_{t+2}+\dots+\gamma^{n-1}r_{t+n}+\gamma^nV(s_{t+n})$$

Intermediate $n$ (often 3 to 10) usually beats both extremes: rewards propagate $n$ states per update rather than one, while variance stays moderate. Rainbow DQN, for example, uses 3-step targets. Rather than pick one $n$, the **$\lambda$-return** averages all of them with geometric weights:

$$G_t^\lambda=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)}$$

$\lambda=0$ gives TD(0) and $\lambda=1$ gives Monte Carlo. The same exponentially weighted average of multi-step errors appears as Generalised Advantage Estimation in lesson 08.06.

## Eligibility traces: the backward view

Computing $G_t^\lambda$ directly requires future rewards. **Eligibility traces** get the same effect online. Each state keeps a trace $e(s)$ that decays by $\gamma\lambda$ per step and is bumped when the state is visited; each TD error then updates all states in proportion to their trace:

```python
def td_lambda_episode(env, V, policy, alpha, gamma, lam):
    e = np.zeros_like(V); s = env.reset(); done = False
    while not done:
        s2, r, done = env.step(policy(s))
        delta = r + (0 if done else gamma * V[s2]) - V[s]
        e *= gamma * lam; e[s] += 1            # accumulating trace
        V += alpha * delta * e                 # credit recent states
        s = s2
```

A reward now gets credited to the state just left at full weight, the one before at $\gamma\lambda$, and so on. For offline updates over a whole episode this matches the $\lambda$-return exactly; "true online TD($\lambda$)" makes them match online too. With function approximation, $e$ becomes a trace over parameter gradients.

| Method | Bias | Variance | Updates | Needs episode end
| --- | --- | --- | --- | --- |
| Monte Carlo | none | high | end of episode | yes
| TD(0) | yes, from bootstrapping | low | every step | no
| n-step TD | less | medium | after n steps | no
| TD($\lambda$) with traces | tunable | tunable | every step | no

<div class="take">
<div class="lab">Key takeaway</div>

model-free prediction moves values towards a target; Monte Carlo uses the full observed return (unbiased, noisy, episode-bound), TD(0) bootstraps from the next state's estimate (biased, low-variance, online), and n-step returns, $\lambda$-returns and eligibility traces span everything in between.

</div>
