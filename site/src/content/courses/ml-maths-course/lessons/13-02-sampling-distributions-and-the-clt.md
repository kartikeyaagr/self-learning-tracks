---
id: "13.02"
course: "ml-maths-course"
module: "13"
slug: "sampling-distributions-and-the-clt"
title: "Sampling distributions and the CLT"
keyIdeas: ["A statistic computed from a random sample is itself a random variable; its distribution over repeated samples is the **sampling distribution**, and its standard deviation is the **standard error**.", "For a mean of $n$ independent draws, $\\mathrm{SE} = \\sigma/\\sqrt n$: halving the error costs four times the data.", "The CLT says the standardised mean tends to $N(0,1)$; Berry-Esseen bounds the error by roughly $0.5\\,\\rho/(\\sigma^3\\sqrt n)$, so skewed data need much larger $n$.", "The CLT fails or is slow with infinite variance (Cauchy, Pareto with $\\alpha\\le 2$), dependent observations, statistics that are not averages (maxima), and tiny samples of skewed data.", "When in doubt, simulate: draw many samples from a plausible population, compute the statistic each time and look at the histogram."]
question: "You measure average session length on 2,000 sessions coming from 200 users (10 sessions each) and compute the SE as s divided by the square root of 2,000. Why is this probably too small, and how would you get a defensible SE?"
answer: "The formula assumes 2,000 independent observations, but sessions from the same user are correlated: a heavy user has consistently long sessions. With positive within-user correlation the variance of the mean is inflated by a design effect of roughly $1 + (m-1)\\rho$ for $m = 10$ sessions per user and intra-user correlation $\\rho$; with $\\rho = 0.3$ that is 3.7, so the true SE is almost twice the naive one and the effective sample size is closer to 540 than 2,000. A defensible SE treats the user as the unit: compute each user's mean (or total sessions and total time, then use the delta method for the ratio), take the SE across the 200 users, or run a bootstrap that resamples users rather than sessions. Session length is also right-skewed, so check that 200 users is enough, or use a log transform or the bootstrap."
sources: [{"title": "Penn State STAT 415: Introduction to Mathematical Statistics", "url": "https://online.stat.psu.edu/stat415/", "note": "Sampling distributions, the CLT and standard errors with derivations."}, {"title": "Wikipedia: Central limit theorem", "url": "https://en.wikipedia.org/wiki/Central_limit_theorem", "note": "Statement, Lyapunov and Lindeberg conditions, and the Berry-Esseen section on convergence rate."}, {"title": "Think Stats (3rd ed.), Allen Downey", "url": "https://allendowney.github.io/ThinkStats/", "note": "Free, simulation-first treatment of sampling distributions in Python."}, {"title": "Seeing Theory (Brown University)", "url": "https://seeing-theory.brown.edu/", "note": "Interactive CLT and sampling visualisations."}]
practice: ["sampling-distribution-of-mean", "central-limit-theorem", "standard-error-comparison", "stratified-sampling"]
tags: ["standard error", "CLT"]
tools: ["NumPy"]
---

## Statistics have distributions too

Lesson 12.01 introduced estimators and their bias and variance. This lesson is the practical side: how big is the noise in a number you computed from data, and when can you treat that noise as normal?

Draw a sample of $n$ values, compute a statistic (mean, median, conversion rate, AUC). Draw a fresh sample and you get a different value. The distribution of that value across all the samples you could have drawn is its **sampling distribution**. Its standard deviation is the **standard error** (SE). The SE is what turns a point estimate into a statement with error bars, and it is what every test in this module divides by.

~~~dg-flow The sampling distribution is a thought experiment you can run in code: repeat the whole data collection, not just the calculation.
Population -(draw n)-> Sample -(compute)-> Statistic -(repeat many times)-> *Sampling distribution -(its std)-> Standard error
~~~

## The standard error of a mean

For $n$ independent draws with variance $\sigma^2$, $\mathrm{Var}(\bar x) = \sigma^2/n$, so

$$\mathrm{SE}(\bar x) = \frac{\sigma}{\sqrt n} \approx \frac{s}{\sqrt n}.$$

Two consequences drive experiment budgets. The error shrinks with $\sqrt n$, so a 2x tighter estimate needs 4x the data. And a proportion is a mean of 0/1 values with $\sigma^2 = p(1-p)$, so $\mathrm{SE}(\hat p) = \sqrt{p(1-p)/n}$.

Other statistics have their own SEs. The median of normal data has $\mathrm{SE} \approx 1.253\,\sigma/\sqrt n$, about 25% noisier than the mean, which is the price of its robustness; on heavy-tailed data the ordering flips and the median wins. When no formula exists, the bootstrap (lesson 13.05) estimates the SE by resampling.

## The central limit theorem

If $x_1,\dots,x_n$ are independent and identically distributed with mean $\mu$ and finite variance $\sigma^2$, then

$$\frac{\bar x - \mu}{\sigma/\sqrt n} \xrightarrow{d} N(0,1).$$

The theorem is about the limit; what matters in practice is the rate. The Berry-Esseen theorem bounds the worst-case gap between the true CDF of the standardised mean and $\Phi$ by $C\rho/(\sigma^3\sqrt n)$ with $\rho = E|x-\mu|^3$ and $C$ below 0.5. Skewness enters through $\rho$: symmetric data are near-normal by $n\approx 30$, while a conversion rate of 0.1% or revenue with a few whales may need thousands of observations before a normal approximation is honest.

```
import numpy as np
rng = np.random.default_rng(0)
pop = rng.lognormal(mean=3, sigma=1.5, size=1_000_000)   # skewed "revenue"
for n in (10, 100, 1000):
    means = rng.choice(pop, size=(5000, n)).mean(axis=1)
    z = (means - pop.mean()) / (pop.std() / np.sqrt(n))
    print(n, round(float(np.mean(z > 1.645)), 3))   # 0.05 if normal
```

The printed tail fraction drifts towards 0.05 as $n$ grows. At small $n$ the sampling distribution of a lognormal mean is itself right-skewed: the right tail is too heavy and the left too light, so one-sided normal tests on the left side are overconfident.

## When the CLT fails

- **Infinite variance.** A Cauchy sample mean has exactly the same distribution as a single observation, for every $n$. Pareto tails with $1 < \alpha \le 2$ have a mean but no variance; averages converge, slowly and to a non-normal stable law. Heavy-tailed revenue and file sizes come close to this regime.
- **Dependence.** For an AR(1) series with lag-1 correlation $\rho$, $\mathrm{Var}(\bar x) \approx \frac{\sigma^2}{n}\cdot\frac{1+\rho}{1-\rho}$. With $\rho = 0.5$ the variance triples, so the effective sample size is $n/3$. Time series, repeated measures on the same user and clustered data (pages within sessions) all break the $\sigma/\sqrt n$ formula.
- **Non-averages.** The maximum, a high quantile like p99.9, or the number of distinct items do not obey the CLT at all; extreme-value theory or the bootstrap (with care) applies instead.
- **Ratio metrics.** Clicks per page view, randomised by user, is a ratio of two correlated means. Treating each page view as independent understates the SE; the delta method or a user-level bootstrap gives the right one.

| Situation | What goes wrong | Fix
| --- | --- | --- |
| Skewed data, small n | Tails of the sampling distribution are asymmetric | Larger n, log transform, bootstrap
| Heavy tails (no variance) | SE formula is meaningless | Median, trimming, winsorizing
| Correlated observations | SE too small, tests overconfident | Cluster-level analysis, block bootstrap
| Ratio of means | Unit of analysis mismatch | Delta method, aggregate per randomisation unit

<div class="ex">
<div class="lab">Worked example</div>

**how precise is a 2% click-through rate?**

With $n = 10{,}000$ independent impressions and $\hat p = 0.02$: $\mathrm{SE} = \sqrt{0.02\cdot0.98/10000} = 0.0014$.

A 95% interval is $0.02 \pm 1.96\cdot0.0014$, roughly 1.73% to 2.27%: a relative error of about $\pm 14\%$.

To get $\pm 5\%$ relative you need $(13.7/5)^2 \approx 7.5$ times as many impressions, about 75,000. Check $n\hat p = 200 \gg 10$, so the normal approximation is fine here; at $p = 0.001$ with the same $n$ it would be shaky.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

the standard error, not the standard deviation, measures how much a statistic would move on fresh data; trust $\sigma/\sqrt n$ and the normal approximation only for independent, finite-variance data with enough observations to overcome the skew.

</div>
