---
id: "08.05"
course: "deep-learning"
module: "08"
slug: "bandits-and-exploration"
title: "Bandits and exploration"
keyIdeas: ["A **multi-armed bandit** is an MDP with one state: each round you pick one of $K$ arms and get a random reward. It isolates the explore-or-exploit trade-off, measured by **regret** $R_T = T\\mu^* - \\sum_t \\mathbb E[\\mu_{a_t}] = \\sum_a \\Delta_a \\mathbb E[N_a(T)]$: how often you played each bad arm times how bad it is.", "Greedy can lock onto a lucky bad arm and fixed-$\\varepsilon$ keeps exploring blindly forever, so both have regret that grows linearly in $T$. The best achievable growth is logarithmic in $T$ (the Lai-Robbins lower bound).", "UCB1 plays $\\arg\\max_a \\hat\\mu_a + \\sqrt{2\\ln t / N_a}$: optimism in the face of uncertainty. Rarely tried arms get a large bonus, and the bonus shrinks like $1/\\sqrt{N_a}$ as an arm is played.", "Thompson sampling keeps a posterior per arm (Beta for yes/no rewards, just two counters), samples one plausible mean per arm and plays the best sample. Each arm is played with the probability that it is best; it is simple and usually the strongest in practice.", "Use a constant step size or discounting when rewards drift. Contextual bandits (LinUCB) add features per round and power recommendation and ad selection; the same optimism bonus reappears in tree search as UCT and PUCT."]
question: "An e-commerce team runs Thompson sampling over 5 page layouts, but conversions are reported in daily batches, so the algorithm picks the layout for every visitor of the day before seeing any of that day's results. They want to switch to UCB1 'because it is deterministic and easier to audit'. What happens to UCB1 under daily batching, and why does Thompson sampling cope better? What would you recommend?"
answer: "UCB1 is deterministic given its counts and estimates. Within a day no results arrive, so the counts do not change, and every visitor that day gets the same layout: the one with the highest upper bound at the start of the day. Exploration then happens one arm per day instead of continuously. With 5 layouts it takes many days just to rotate through the arms a few times, and a whole day of traffic can be spent on a layout that a few hundred visitors would have ruled out. Exploitation suffers too: even once a layout is clearly best, UCB1 may hand an entire day to an under-sampled rival because its bonus is temporarily the largest. Regret therefore grows roughly in proportion to the batch size. Thompson sampling draws a fresh posterior sample per visitor, so within one day traffic is spread across layouts in proportion to the probability that each is best given yesterday's data: mostly the likely winner, some share for plausible rivals, almost none for clearly bad layouts. That is natural, well-calibrated batch exploration, and it is why Thompson sampling is known to tolerate delayed feedback well. Recommendation: keep Thompson sampling and make it auditable instead. Use a seeded random generator, log for every visitor the layout shown, the posterior parameters and the probability with which that layout was chosen (estimate it by repeated sampling if needed). The logged probabilities also allow unbiased offline evaluation of other policies with inverse propensity weighting. If a deterministic rule is mandatory, use a batched UCB variant that splits each day's traffic across several arms, and document the batch size it was designed for."
sources: [{"title": "Lattimore and Szepesvari: Bandit Algorithms", "url": "https://tor-lattimore.com/downloads/book/book.pdf", "note": "Free textbook. Chapters 6 to 8 (explore-then-commit, UCB) and 36 (Thompson sampling); chapter 16 for lower bounds."}, {"title": "Russo et al.: A Tutorial on Thompson Sampling", "url": "https://arxiv.org/abs/1707.02038", "note": "Beta-Bernoulli Thompson sampling, its extensions and when it beats UCB."}, {"title": "Li et al.: A Contextual-Bandit Approach to Personalized News Article Recommendation", "url": "https://arxiv.org/abs/1003.0146", "note": "LinUCB and offline evaluation on Yahoo! front-page data."}]
practice: ["ucb1-for-multi-armed-bandits", "thompson-sampling-beta-bernoulli", "epsilon-greedy-action-selection", "softmax-boltzmann-exploration", "linucb-contextual-bandit"]
tags: ["bandits", "UCB", "Thompson sampling"]
tools: ["NumPy"]
---

## What you will learn

- What a **multi-armed bandit** is, and why it is the cleanest place to study the explore-or-exploit dilemma.
- How to score a strategy with **regret**, and why "logarithmic" regret is the best any strategy can do.
- Why greedy and fixed-$\varepsilon$ strategies waste reward forever, with a simulation that shows it.
- How **UCB1** (optimism plus a shrinking bonus) and **Thompson sampling** (act on a random plausible guess) explore in a directed way.
- How the same ideas extend to non-stationary rewards, contextual bandits (LinUCB) and tree search.

*Before you start:* you should know $\varepsilon$-greedy action selection from lesson 08.04 and the incremental (TD-style) update from lesson 08.03. A little probability helps: averages, the idea of a confidence interval, and what a probability distribution over an unknown number means.

## From Q-learning to the simplest RL problem

Lesson 08.04 bolted $\varepsilon$-greedy onto Q-learning and decayed $\varepsilon$ by hand. That hides a real question: *how much* should an agent explore, and *where*? In a full MDP this is tangled up with states and delayed rewards.

A bandit strips all of that away. There is one state. Each round you pull one of $K$ arms (the name comes from slot machines, "one-armed bandits") and get a random reward. Your choice has no effect on the future except through what it teaches you. Everything that is left is the dilemma itself: **exploit** the arm that looks best, or **explore** an arm you know less about.

Bandits also run adaptive A/B tests, ad and recommendation selection, clinical dose finding and hyperparameter search, and their rules reappear inside Monte Carlo tree search (lesson 09.02).

## The idea in one picture

You are emailing a newsletter and have three subject lines, A, B and C. Each has a true open rate that you cannot see: 0.20, 0.25 and 0.30. You send each subject line to 20 people first:

| Subject line | Sent | Opened | Estimated open rate | True rate (hidden)
| --- | --- | --- | --- | --- |
| A | 20 | 4 | 0.20 | 0.20
| B | 20 | 7 | 0.35 | 0.25
| C | 20 | 5 | 0.25 | 0.30

Line B looks best, but only by luck. C is truly best. The figure shows why you cannot tell yet: each curve says how plausible each open rate is for one subject line given its 20 results, and the curves overlap heavily.

~~~dg-fig 08.05-idea.svg
After 20 sends each, B looks best (peak near 0.35) but the curves overlap a lot; C, the truly best line (dashed green), is still entirely plausible.
~~~

Drawing many plausible rates from these three curves, B comes out on top 67% of the time, C 22% and A 11%. A **greedy** agent that always picks the highest estimate would send B to everyone from now on and, never sending C again, never discover its mistake. A good strategy keeps giving C some traffic until the evidence settles the question, and stops wasting traffic on A once A is clearly worse.

Three terms for the rest of the lesson:

- An **arm** is one option; each has an unknown mean reward $\mu_a$ (here, the open rate).
- **Exploitation** means playing the arm with the best current estimate, to collect reward now.
- **Exploration** means playing an arm whose value is uncertain, to collect information that pays off later.

## Step 1: measure a strategy with regret

To score a strategy we count what it lost compared to an oracle that knew the best arm from the start. That loss is the **regret** after $T$ rounds:

$$R_T = T\mu^* - \sum_{t=1}^{T}\mathbb E[\mu_{a_t}] = \sum_a \Delta_a\,\mathbb E[N_a(T)],\qquad \Delta_a = \mu^* - \mu_a.$$

Here $\mu^*$ is the best arm's mean, $a_t$ is the arm played at round $t$, $N_a(T)$ is how many times arm $a$ was played in $T$ rounds, and $\Delta_a$ is the **gap**: how much worse arm $a$ is than the best. The second form is the useful one. Regret is "how often you played each bad arm" times "how bad it is". The best arm has gap 0, so pulling it costs nothing.

<div class="ex">
<div class="lab">Worked example</div>

**regret from pull counts**

In the simulation later in this lesson, $\varepsilon$-greedy with $\varepsilon=0.1$ played 5,000 rounds and pulled A, B and C on average 286, 622 and 4,092 times.

Gaps: $\Delta_A = 0.30 - 0.20 = 0.10$, $\Delta_B = 0.30 - 0.25 = 0.05$, $\Delta_C = 0$.

Regret: $0.10 \times 286 + 0.05 \times 622 + 0 \times 4092 = 28.6 + 31.1 = 59.7$ lost opens.

The oracle would have collected $5000 \times 0.30 = 1500$ opens; this strategy collected about $1500 - 59.7 \approx 1440$.

</div>

How fast can regret grow? Any strategy that never stops exploring at a fixed rate pays a fixed fraction of every round, so its regret grows in a straight line: **linear regret**. Lai and Robbins (1985) proved the other end: any strategy that works well on every problem must play each bad arm at least about $\ln T / \mathrm{KL}(\mu_a\|\mu^*)$ times, where KL measures how hard the two arms are to tell apart. So regret must grow at least like $\ln T$. Good algorithms reach that rate, which means that the fraction of rounds wasted goes to zero.

## Step 2: estimate each arm's value

Every strategy keeps an estimate $\hat\mu_a$ per arm, normally the average reward seen so far. You do not need to store every reward; update the average in place:

$$\hat\mu_a \leftarrow \hat\mu_a + \frac{1}{N_a}\,(r - \hat\mu_a).$$

Here $r$ is the reward just received and $N_a$ is the pull count including this pull. In words: move the estimate toward the new reward by a fraction $1/N_a$ of the surprise. This is the same shape as the TD update of lesson 08.03, with step size $1/N_a$.

Number check: B's estimate is $7/20 = 0.35$. The 21st send is opened ($r=1$): $0.35 + (1 - 0.35)/21 = 0.381$, which is exactly $8/21$.

If rates drift, replace $1/N_a$ with a constant step size $\alpha$ so old rewards are forgotten exponentially (see "What goes wrong").

## Step 3: greedy, epsilon-greedy and softmax

**Greedy** always plays $\arg\max_a \hat\mu_a$. As the opening example showed, an unlucky start can lock it onto a bad arm forever. In our simulation greedy ended with C as its most-played arm in only 26% of the 400 runs.

**$\varepsilon$-greedy** plays a uniformly random arm with probability $\varepsilon$ and the greedy arm otherwise. It can no longer lock in, but its exploration is blind. With fixed $\varepsilon$ it pulls each bad arm about $\varepsilon T / K$ times no matter how much evidence has piled up, so regret is linear: with $\varepsilon = 0.1$, $K=3$ and our gaps it loses about $0.1 \times (0.10 + 0.05)/3 = 0.005$ opens every round, forever. Decaying $\varepsilon_t \propto 1/t$ restores logarithmic regret in theory, but the right constant depends on the unknown gaps.

**Softmax** (Boltzmann) exploration picks arm $a$ with probability proportional to $\exp(\hat\mu_a/\tau)$, so near-best arms get more exploration than hopeless ones. The temperature $\tau$ plays the role of $\varepsilon$ and also needs tuning.

## Step 4: UCB1, optimism in the face of uncertainty

The weakness of $\varepsilon$-greedy is that it ignores *how much* it knows about each arm. UCB1 (Auer, Cesa-Bianchi and Fischer, 2002) uses that directly. For each arm it computes an optimistic value, the highest mean that is still plausible given the data, and plays the arm whose optimistic value is highest:

$$a_t = \arg\max_a\ \hat\mu_a + \sqrt{\frac{2\ln t}{N_a}}.$$

The first term is the estimate. The second is an **exploration bonus**: $t$ is the current round and $N_a$ the number of times arm $a$ has been played. Each arm is played once first so that $N_a > 0$. This rule is called **optimism in the face of uncertainty**: treat every arm as if it were as good as it could plausibly be. If an optimistic arm turns out to be good, you gain; if it turns out to be bad, its bonus shrinks and you stop.

An arm is played either because its estimate is high (exploitation) or because its bonus is large (exploration). The bonus shrinks like $1/\sqrt{N_a}$ when the arm is played and grows slowly, like $\sqrt{\ln t}$, while it is ignored, so no arm is abandoned for good.

<div class="ex">
<div class="lab">Worked example</div>

**one UCB1 decision**

At $t = 100$: arm A has $\hat\mu = 0.60$ from 80 pulls; arm B has $\hat\mu = 0.50$ from 20 pulls. First $\ln 100 = 4.605$, so $2\ln t = 9.21$.

A: bonus $\sqrt{9.21/80} = \sqrt{0.115} = 0.34$, score $0.60 + 0.34 = 0.94$.

B: bonus $\sqrt{9.21/20} = \sqrt{0.461} = 0.68$, score $0.50 + 0.68 = 1.18$.

UCB1 plays B. With only 20 samples, a true mean above 0.6 is still plausible for B. B has a quarter of A's pulls, so its bonus is $\sqrt 4 = 2$ times larger.

Each further pull of B shrinks its bonus. Unless B's results turn out better than 0.6, A wins the comparison again within a few rounds.

</div>

~~~dg-fig 08.05-ucb.svg
Left: the worked example as estimate plus bonus; the less-tried arm B wins on its larger bonus. Right: the bonus falls quickly with pulls and rises only slowly with the round number t.
~~~

UCB1 is deterministic, needs no tuning, and for rewards in $[0,1]$ its expected regret is at most about $\sum_a 8\ln T/\Delta_a$ plus a constant. That is logarithmic, but in practice UCB1 explores more than it needs to, as the simulation shows.

## Step 5: Thompson sampling, act on a plausible guess

Thompson sampling (proposed by Thompson in 1933, rediscovered for practice around 2010) is the Bayesian answer. Keep a **posterior**, a probability distribution over each arm's unknown mean, as in the opening figure. Each round:

1. Draw one random sample $\tilde\theta_a$ from each arm's posterior.
2. Play the arm with the largest sample.
3. Observe the reward and update that arm's posterior.

For yes/no rewards (opened or not) the **Beta distribution** makes this trivial. Start every arm at $\text{Beta}(1, 1)$, which is uniform on $[0,1]$. After $s$ successes and $f$ failures the posterior is $\text{Beta}(1+s, 1+f)$, so the whole "model" is two counters per arm.

Number check: B with 7 opens out of 20 has posterior $\text{Beta}(8, 14)$, whose mean is $8/22 = 0.36$. Its width (standard deviation about 0.10) is what makes the overlap in the opening figure.

Each arm is played with exactly the probability that it is best given the data (**probability matching**). An uncertain arm has a wide posterior, so its samples sometimes come out high and it gets tried; a clearly worse arm's samples almost never win. Exploration is spent where the answer is still in doubt.

~~~dg-fig 08.05-posterior.svg
One Thompson sampling run. After 20 rounds all three curves are wide. By 2,000 rounds C (the best) has 1,713 pulls and a sharp peak, while A was tried only 16 times: once its curve sat clearly to the left, its samples stopped winning.
~~~

Thompson sampling matches the Lai-Robbins rate for Bernoulli arms, usually beats UCB1 in practice, and copes well with delayed or batched feedback because its randomness spreads each batch across arms.

~~~dg-layers From blind to directed exploration; regret on a typical Bernoulli problem improves going down.
Greedy | never explores; can lock onto a bad arm, linear regret
$\varepsilon$-greedy | explores uniformly at random; linear regret unless decayed
UCB1 | explores arms whose upper bound is high; $O(\log T)$ regret
*Thompson sampling | explores in proportion to the probability of being best; $O(\log T)$, strong in practice
~~~

## The four strategies in code

This NumPy script plays 400 independent games of 5,000 rounds at once (one row per game) on the newsletter problem and reports the average regret.

```python
import numpy as np

def simulate(policy, mu, T=5000, runs=400, eps=0.1, seed=0):
    # play `runs` independent bandit games at once; return mean cumulative regret
    rng = np.random.default_rng(seed)
    K = len(mu); rows = np.arange(runs)
    N = np.zeros((runs, K))            # pulls per arm
    S = np.zeros((runs, K))            # total reward per arm
    regret = np.zeros(T)
    for t in range(T):
        Q = S / np.maximum(N, 1)       # sample-average estimates
        if policy == "thompson":       # Beta(1, 1) prior -> Beta(1+S, 1+N-S) posterior
            a = rng.beta(S + 1, N - S + 1).argmax(1)
        elif t < K:                    # others: play each arm once first
            a = np.full(runs, t)
        elif policy == "greedy":
            a = Q.argmax(1)
        elif policy == "eps":
            explore = rng.random(runs) < eps
            a = np.where(explore, rng.integers(0, K, runs), Q.argmax(1))
        elif policy == "ucb1":
            a = (Q + np.sqrt(2 * np.log(t + 1) / N)).argmax(1)
        r = (rng.random(runs) < mu[a]).astype(float)   # Bernoulli reward
        N[rows, a] += 1; S[rows, a] += r
        regret[t] = (mu.max() - mu[a]).mean()
    return regret.cumsum(), N.mean(0)

mu = np.array([0.20, 0.25, 0.30])      # true open rates, hidden from the agent
for p in ["greedy", "eps", "ucb1", "thompson"]:
    reg, pulls = simulate(p, mu)
    print(f"{p:9s} regret@1000={reg[999]:6.1f}  regret@5000={reg[-1]:6.1f}  pulls={pulls.round()}")
# greedy    regret@1000=  65.3  regret@5000= 326.3  pulls=[2800.  926. 1274.]
# eps       regret@1000=  25.4  regret@5000=  59.7  pulls=[ 286.  622. 4092.]
# ucb1      regret@1000=  34.9  regret@5000= 114.9  pulls=[ 580. 1136. 3283.]
# thompson  regret@1000=  19.9  regret@5000=  35.7  pulls=[ 139.  436. 4425.]
```

Note that greedy's average pull counts favour A, the *worst* arm: in many runs A happened to score early and was never questioned again.

## Reading the result

~~~dg-fig 08.05-regret.svg
Left: over 5,000 rounds greedy's regret is a straight line, and Thompson sampling is lowest. Right (log time axis, 200,000 rounds): ε-greedy's linear cost eventually overtakes UCB1, while Thompson sampling stays far below both.
~~~

Three lessons from the curves:

- **Shape matters more than height early on.** A straight line (greedy, and $\varepsilon$-greedy in the long run) means a fixed fraction of every round is wasted forever. A curve that bends over means the strategy is converging. Plotted against $\log t$, logarithmic regret becomes a straight line and linear regret shoots upward, which is the clearest way to tell them apart.
- **Finite horizons can flip rankings.** Over 5,000 rounds, $\varepsilon$-greedy (59.7) beats UCB1 (114.9) because UCB1's bonus is conservative. In our 200,000-round run $\varepsilon$-greedy passes UCB1 at around $t \approx 70{,}000$ and ends at 1,043 versus 520. If your horizon is short, measure on a simulation of your own problem rather than trusting asymptotic rates.
- **Thompson sampling wins at both horizons here** (35.7 at 5,000 rounds and 71 at 200,000).

In a real system, log pulls per arm over time, each estimate with its uncertainty, and the probability with which each action was chosen (this enables offline evaluation later).

## What goes wrong

**Non-stationary rewards.** Sample averages weight a reward from last month the same as one from today. In the next figure the arms' rates swap at round 3,000: arm A jumps from 0.20 to 0.30 and C drops to 0.20.

~~~dg-fig 08.05-drift.svg
Left: after the swap the sample-average estimate of A barely moves, while a constant step size climbs to the new rate. Right: the sample-average agent keeps playing the old favourite; the constant-step agent recovers within a few hundred rounds, at the cost of a noisier estimate before the swap.
~~~

After round 4,000 the sample-average agent put only 4% of its pulls on the new best arm, the constant-step agent 86%. The price of forgetting is noise: before the swap the constant-step agent sometimes wandered away from the best arm. Use a constant step, a sliding window, or discount old counts in Thompson sampling when the world drifts.

| Symptom | Likely cause | Fix
| --- | --- | --- |
| One arm gets nearly all traffic from the first day | greedy or tiny $\varepsilon$ locked in on early luck | use UCB or Thompson sampling; or optimistic initial estimates
| Bad arms keep a steady share of traffic for months | fixed $\varepsilon$ or fixed temperature | decay $\varepsilon$, or switch to directed exploration
| Performance decays after a good start | rewards drift (seasonality, novelty wearing off) | constant step size, sliding window, discounted posteriors
| UCB sends a whole day of traffic to one arm | deterministic rule with batched feedback | Thompson sampling, or a batched UCB variant
| Results look great but cannot be reproduced offline | action probabilities were not logged | log the propensity of every action
| UCB explores almost forever, or almost never | reward scale far from $[0,1]$, so the bonus is too large or too small | rescale rewards to $[0,1]$, or use a variance-aware bonus

## Going deeper

**Where the UCB bonus comes from.** For rewards in $[0,1]$, Hoeffding's inequality says the average of $N$ samples falls more than $\epsilon$ below the true mean with probability at most $e^{-2N\epsilon^2}$. Set that probability to $t^{-4}$ and solve for $\epsilon$: $\epsilon = \sqrt{2\ln t / N}$, exactly the UCB1 bonus. So $\hat\mu_a + \text{bonus}$ is an upper confidence bound that fails with a probability that shrinks fast as $t$ grows. A suboptimal arm can only win once its bound has dropped below $\mu^*$, which needs $N_a \gtrsim 8\ln T/\Delta_a^2$ pulls; multiplying by $\Delta_a$ gives the $8\ln T/\Delta_a$ regret term.

**The lower bound, with numbers.** For Bernoulli arms Lai and Robbins give $\liminf_T R_T/\ln T \ge \sum_{a:\Delta_a>0} \Delta_a/\mathrm{KL}(\mu_a\|\mu^*)$, with $\mathrm{KL}(p\|q) = p\ln\frac pq + (1-p)\ln\frac{1-p}{1-q}$. Here $\mathrm{KL}(0.20\|0.30) = 0.026$ and $\mathrm{KL}(0.25\|0.30) = 0.0062$. B is four times harder to rule out than A, because its rate is close to C's. The coefficient is $0.10/0.026 + 0.05/0.0062 \approx 12$, so regret must eventually grow at least like $12\ln T$. This is an asymptotic statement: at finite $T$ a good algorithm can sit below $12\ln T$ (Thompson sampling did, at 71 after 200,000 rounds versus $12\ln 200{,}000 \approx 146$), but no algorithm can stay below it as $T \to \infty$ on every problem. Gap-free (worst-case) bounds are $\Theta(\sqrt{KT})$.

**Regret versus best-arm identification.** A classic A/B test wants to name the best arm with confidence, not to earn reward while learning. Thompson sampling starves bad arms of data, which is good for regret but leaves wide intervals on those arms.

**Contextual bandits.** Often each round comes with features $x_t$ (user, page, time of day) and the best arm depends on them. **LinUCB** (Li et al., 2010) models arm $a$'s reward as linear, $\mu_a(x) = \theta_a^\top x$. It keeps $A_a = I + \sum x x^\top$ and $b_a = \sum r\,x$ over the rounds where $a$ was played, estimates $\hat\theta_a = A_a^{-1}b_a$ (ridge regression), and plays

$$\arg\max_a\ \hat\theta_a^\top x + \alpha\sqrt{x^\top A_a^{-1}x}.$$

The bonus is large in feature directions where arm $a$ has little data, the matrix version of $1/\sqrt{N_a}$. Li et al. used it for news article recommendation on the Yahoo! front page. Thompson sampling extends the same way by sampling $\theta_a$ from a Gaussian posterior.

**From bandits to RL and search.** A contextual bandit is RL with episodes of length one; full RL must also plan exploration across time, and count-based bonuses descend directly from UCB. In tree search every node is a bandit over moves: UCT applies UCB1 at each node, and AlphaGo Zero's PUCT adds a policy-network prior to the bonus (lesson 09.02).

## Key terms

| Term | Meaning
| --- | --- |
| Multi-armed bandit | Repeatedly choose one of $K$ options with unknown reward distributions; no state
| Exploration / exploitation | Trying uncertain options to learn vs. playing the best-looking option to earn
| Regret | Expected reward lost versus always playing the best arm: $\sum_a \Delta_a\,\mathbb E[N_a]$
| Gap $\Delta_a$ | How much worse arm $a$'s mean is than the best arm's
| $\varepsilon$-greedy | Random arm with probability $\varepsilon$, else the best estimate; linear regret if $\varepsilon$ is fixed
| UCB1 | Play the arm with the highest estimate plus bonus $\sqrt{2\ln t/N_a}$
| Thompson sampling | Sample a mean from each arm's posterior and play the best sample
| Beta posterior | $\text{Beta}(1+s, 1+f)$ after $s$ successes and $f$ failures from a uniform prior
| Contextual bandit / LinUCB | Bandit with features each round; LinUCB adds a ridge-regression uncertainty bonus

<div class="take">
<div class="lab">Key takeaway</div>

bandits isolate exploration; regret must grow at least like $\log T$, which UCB (optimistic bonuses that shrink with $1/\sqrt{N}$) and Thompson sampling (sample from the posterior, act greedily on the sample) achieve, while greedy and fixed-$\varepsilon$ strategies waste a fixed fraction of pulls forever.

</div>
