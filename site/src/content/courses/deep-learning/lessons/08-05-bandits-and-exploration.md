---
id: "08.05"
course: "deep-learning"
module: "08"
slug: "bandits-and-exploration"
title: "Bandits and exploration"
keyIdeas: ["A multi-armed bandit is an MDP with one state: pick one of $K$ arms, get a random reward. It isolates the exploration-exploitation trade-off, measured by regret $R_T=T\\mu^*-\\sum_t\\mathbb E[\\mu_{a_t}]$.", "Greedy and fixed-$\\varepsilon$ strategies have regret linear in $T$; the best achievable is logarithmic in $T$ (the Lai-Robbins lower bound).", "UCB1 plays $\\arg\\max_a\\hat\\mu_a+\\sqrt{2\\ln t/N_a}$: optimism in the face of uncertainty, with a bonus that shrinks as an arm is tried.", "Thompson sampling keeps a posterior per arm (Beta for Bernoulli rewards), samples one plausible mean per arm and plays the best sample; it is simple, Bayesian and usually the strongest in practice.", "Contextual bandits (LinUCB) add features per round; they power recommendation and ad selection, and the same optimism bonus reappears in tree search as PUCT."]
question: "An e-commerce team runs Thompson sampling over 5 page layouts, but conversions are reported in daily batches, so the algorithm picks the layout for every visitor of the day before seeing any of that day's results. They want to switch to UCB1 'because it is deterministic and easier to audit'. What happens to UCB1 under daily batching, and why does Thompson sampling cope better?"
answer: "UCB1 is deterministic given its counts. Within a day the counts do not change, so every visitor that day gets the same layout: the one with the highest upper bound at the start of the day. Exploration then happens one arm per day instead of continuously, so it takes many days to try the arms enough, and a whole day of traffic can be spent on an arm that a few hundred visitors would have ruled out. Regret grows with batch size. Thompson sampling draws a fresh posterior sample per visitor, so within one day traffic is spread across layouts in proportion to the probability that each is best, which is natural, well-calibrated batch exploration. If auditability matters, keep Thompson sampling with logged seeds and per-visitor assignment probabilities (which also enables unbiased offline evaluation), or use a batched UCB variant that spreads each batch across arms."
sources: [{"title": "Lattimore and Szepesvari: Bandit Algorithms", "url": "https://tor-lattimore.com/downloads/book/book.pdf", "note": "Free textbook. Chapters 6 to 8 (explore-then-commit, UCB) and 36 (Thompson sampling); chapter 16 for lower bounds."}, {"title": "Russo et al.: A Tutorial on Thompson Sampling", "url": "https://arxiv.org/abs/1707.02038", "note": "Beta-Bernoulli Thompson sampling, its extensions and when it beats UCB."}, {"title": "Li et al.: A Contextual-Bandit Approach to Personalized News Article Recommendation", "url": "https://arxiv.org/abs/1003.0146", "note": "LinUCB and offline evaluation on Yahoo! front-page data."}]
practice: ["ucb1-for-multi-armed-bandits", "thompson-sampling-beta-bernoulli", "epsilon-greedy-action-selection", "softmax-boltzmann-exploration", "linucb-contextual-bandit"]
tags: ["bandits", "UCB", "Thompson sampling"]
tools: ["NumPy"]
---

## The simplest RL problem

In a **multi-armed bandit**, each round you choose one of $K$ arms and receive a reward drawn from that arm's unknown distribution with mean $\mu_a$. There is no state, so actions have no long-term consequences except what you learn from them. That isolates the hardest question in RL: should you **exploit** the arm that looks best now or **explore** others that might be better? Bandits model A/B tests that adapt while running, ad and recommendation selection, clinical dose finding and hyperparameter search, and their exploration rules reappear inside MCTS (lesson 09.02).

Performance is measured by **regret**, the expected reward lost relative to always playing the best arm $\mu^*$:

$$R_T=T\mu^*-\sum_{t=1}^{T}\mathbb E[\mu_{a_t}]=\sum_a\Delta_a\,\mathbb E[N_a(T)],\qquad \Delta_a=\mu^*-\mu_a$$

Regret is the sum over bad arms of how often you play them times how bad they are. Lai and Robbins showed that any reasonable algorithm must play each suboptimal arm about $\ln T/\mathrm{KL}(\mu_a\|\mu^*)$ times, so regret must grow at least logarithmically in $T$. Good algorithms achieve $O(\log T)$; naive ones grow linearly.

## Estimating arm values

Each arm's estimate is a running mean, updated incrementally: $\hat\mu_a\leftarrow\hat\mu_a+\frac{1}{N_a}(r-\hat\mu_a)$. For non-stationary rewards, replace $1/N_a$ by a constant step size so old rewards are forgotten exponentially, exactly the TD-style update of lesson 08.03.

## Epsilon-greedy and softmax

**$\varepsilon$-greedy** plays a random arm with probability $\varepsilon$ and the best estimate otherwise. With fixed $\varepsilon$, it plays each bad arm about $\varepsilon T/K$ times forever, so regret is linear; decaying $\varepsilon_t\propto1/t$ fixes the rate but needs tuning to the unknown gaps. It also explores blindly: an arm known to be terrible gets as many random pulls as a promising one. **Softmax (Boltzmann)** exploration picks arm $a$ with probability $\propto\exp(\hat\mu_a/\tau)$, which at least explores near-best arms more; the temperature $\tau$ plays the role of $\varepsilon$.

## UCB: optimism in the face of uncertainty

**UCB1** (Auer et al., 2002) plays the arm with the highest upper confidence bound:

$$a_t=\arg\max_a\ \hat\mu_a+\sqrt{\frac{2\ln t}{N_a}}$$

The bonus comes from Hoeffding's inequality: with high probability the true mean lies below the bound. An arm is played either because its estimate is high (exploitation) or because it has been tried rarely (exploration), and the bonus shrinks like $1/\sqrt{N_a}$. Each arm is played once first. UCB1 has $O(\sum_a\ln T/\Delta_a)$ regret for rewards in $[0,1]$, deterministic behaviour and no tuning.

<div class="ex">
<div class="lab">Worked example</div>

**one UCB1 decision**

At $t=100$: arm A has $\hat\mu=0.60$ from 80 pulls; arm B has $\hat\mu=0.50$ from 20 pulls. $\ln100=4.61$.

A: $0.60+\sqrt{9.21/80}=0.60+0.34=0.94$. B: $0.50+\sqrt{9.21/20}=0.50+0.68=1.18$.

UCB1 plays B: with only 20 samples, a true mean above 0.6 is still plausible. Each further pull of B shrinks its bonus until A wins again, unless B turns out better.

</div>

## Thompson sampling

**Thompson sampling** (1933, rediscovered around 2010) is Bayesian: keep a posterior over each arm's mean, draw one sample from each posterior, play the arm with the highest sample. For Bernoulli rewards (click or no click) the Beta prior is conjugate, so the update is two counters per arm:

```python
import numpy as np
def thompson(successes, failures, rng):          # arrays of length K, Beta(1, 1) prior
    theta = rng.beta(successes + 1, failures + 1)
    return int(np.argmax(theta))
# after playing arm a and seeing reward r in {0, 1}:
#   successes[a] += r; failures[a] += 1 - r
```

An arm is played with exactly the posterior probability that it is the best arm (probability matching). Uncertain arms produce wide samples and sometimes win; clearly bad arms almost never do. Thompson sampling achieves the Lai-Robbins rate for Bernoulli arms, handles delayed or batched feedback gracefully because its randomisation avoids playing the same arm for a whole batch, and usually beats UCB empirically.

~~~dg-layers From blind to directed exploration; regret on a typical Bernoulli problem improves going down.
Greedy | never explores; can lock onto a bad arm, linear regret
$\varepsilon$-greedy | explores uniformly at random; linear regret unless decayed
UCB1 | explores arms whose upper bound is high; $O(\log T)$ regret
*Thompson sampling | explores in proportion to the probability of being best; $O(\log T)$, strong in practice
~~~

## Contextual bandits and beyond

In a **contextual bandit** each round comes with features $x_t$ (user, page, time), and the best arm depends on them. **LinUCB** models each arm's reward as linear, $\mu_a(x)=\theta_a^\top x$, fits $\theta_a$ by ridge regression, and adds a bonus $\alpha\sqrt{x^\top A_a^{-1}x}$, the uncertainty of the prediction in direction $x$. Li et al. (2010) used it for Yahoo! news article recommendation. Contextual bandits are one-step RL; full RL adds the fact that actions change future states, so exploration must also be planned over time. The same optimism idea drives UCT and AlphaGo Zero's PUCT rule, where each tree node is a bandit over moves (lesson 09.02).

<div class="take">
<div class="lab">Key takeaway</div>

bandits isolate exploration; regret must grow at least like $\log T$, which UCB (optimistic bonuses that shrink with $1/\sqrt{N}$) and Thompson sampling (sample from the posterior, act greedily on the sample) achieve, while greedy and fixed-$\varepsilon$ strategies waste a fixed fraction of pulls forever.

</div>
