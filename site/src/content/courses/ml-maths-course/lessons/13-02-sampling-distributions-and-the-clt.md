---
id: "13.02"
course: "ml-maths-course"
module: "13"
slug: "sampling-distributions-and-the-clt"
title: "Sampling distributions and the CLT"
keyIdeas: ["A statistic computed from a random sample is itself random. Its distribution over repeated samples is the **sampling distribution**, and its standard deviation is the **standard error**: how much the number would move if you collected the data again.", "For a mean of $n$ independent draws, $\\mathrm{SE} = \\sigma/\\sqrt n \\approx s/\\sqrt n$; for a proportion, $\\sqrt{p(1-p)/n}$. Halving the error costs four times the data.", "The central limit theorem says the standardised mean tends to $N(0,1)$ whatever the data look like, but the speed depends on skew: symmetric data are close by $n \\approx 30$, while heavily skewed revenue-like data can still give wrong 95% intervals at thousands of observations.", "The plain formula and the bell curve fail with infinite variance (Cauchy, Pareto with $\\alpha \\le 2$), dependent or clustered observations, statistics that are not averages (maxima, p99.9), and ratio metrics where the unit of analysis is not the unit of randomisation.", "When in doubt, simulate: draw many samples from a plausible population, compute the statistic each time, and check the spread and how often your interval misses the truth."]
question: "You measure average session length on 2,000 sessions coming from 200 users (10 sessions each) and compute the SE as s divided by the square root of 2,000. Why is this probably too small, and how would you get a defensible SE?"
answer: "The formula assumes 2,000 independent observations, but sessions from the same user are correlated: a heavy user has consistently long sessions, a casual user consistently short ones. With positive within-user correlation the variance of the mean is inflated by a design effect of roughly $1 + (m-1)\\rho$ for $m = 10$ sessions per user and intra-user correlation $\\rho$. With $\\rho = 0.3$ that is $1 + 9 \\times 0.3 = 3.7$, so the true SE is about $\\sqrt{3.7} \\approx 1.9$ times the naive one, and the effective sample size is about $2000/3.7 \\approx 540$ rather than 2,000. A defensible SE treats the user as the unit: compute each user's mean session length and take the SE across the 200 users ($s_{\\text{users}}/\\sqrt{200}$), or, if you want the session-weighted mean, treat it as a ratio (total time over total sessions per user) and use the delta method, or run a bootstrap that resamples whole users rather than individual sessions. Session length is also right-skewed, so check that 200 users is enough for the normal approximation by looking at the histogram of user means or by bootstrapping; a log transform or a median may be more stable if a few users dominate."
sources: [{"title": "Penn State STAT 415: Introduction to Mathematical Statistics", "url": "https://online.stat.psu.edu/stat415/", "note": "Sampling distributions, the CLT and standard errors with derivations."}, {"title": "Wikipedia: Central limit theorem", "url": "https://en.wikipedia.org/wiki/Central_limit_theorem", "note": "Statement, Lyapunov and Lindeberg conditions, and the Berry-Esseen section on convergence rate."}, {"title": "Think Stats (3rd ed.), Allen Downey", "url": "https://allendowney.github.io/ThinkStats/", "note": "Free, simulation-first treatment of sampling distributions in Python."}, {"title": "Seeing Theory (Brown University)", "url": "https://seeing-theory.brown.edu/", "note": "Interactive CLT and sampling visualisations."}]
practice: ["sampling-distribution-of-mean", "central-limit-theorem", "standard-error-comparison", "stratified-sampling"]
tags: ["standard error", "CLT"]
tools: ["NumPy"]
---

## What you will learn

- What a **sampling distribution** is: the spread of answers you would get if you repeated the whole data collection, not just the calculation.
- How to compute the **standard error** of a mean and of a proportion, and why halving it costs four times the data.
- What the central limit theorem (CLT) promises, and how to tell from the data how large $n$ has to be before that promise holds.
- How to check all of it by simulation in NumPy.
- When $\sigma/\sqrt n$ and the bell curve mislead: heavy tails, dependence, non-averages and ratio metrics.

*Before you start:* you need the mean, variance and standard deviation from lesson 13.01, and the idea of a random variable and its expectation (module 12, in particular lesson 12.01 on estimators and their bias and variance). Knowing the normal distribution's 68/95 rule helps.

Lesson 13.01 described a single dataset. This lesson asks a different question: the number you computed (an average session length, a click-through rate, an AUC) came from one sample. If you collected the data again, how much would that number move? The answer turns "the mean is 7.4 minutes" into "7.4 minutes, give or take 1.7", and it is the denominator of every test in this module.

## The idea in one picture

Suppose session lengths on your site follow the left panel of the figure: most sessions are a few minutes long, a few run past half an hour, and the true average is 6 minutes with a standard deviation of 6 minutes. You never see this population, only a study: 25 sessions logged one afternoon. Here are five such studies, each drawn independently from the same population (simulated):

| Study | 1 | 2 | 3 | 4 | 5
| --- | --- | --- | --- | --- | --- |
| Average of 25 sessions (min) | 7.75 | 5.78 | 6.15 | 5.47 | 5.13
| Sample standard deviation $s$ (min) | 5.67 | 5.86 | 6.28 | 5.11 | 5.36

Same population, same method, five different averages, ranging from 5.1 to 7.8. Nothing went wrong; this is what sampling does. Now repeat the study 20,000 times and draw a histogram of the averages.

~~~dg-fig 13.02-idea.svg
Left: individual sessions are strongly right-skewed. Right: the averages of 25 sessions pile up around the true mean of 6, with a standard deviation of only 1.2, and the pile already looks close to a bell curve.
~~~

Three things to notice:

- The averages centre on the true mean, 6. The sample mean is **unbiased**: it is right on average.
- They vary far less than single sessions: standard deviation 1.2 instead of 6, five times less.
- Their histogram is close to a normal curve, even though the sessions themselves look nothing like one.

The histogram on the right is the **sampling distribution** of the mean: the distribution of a statistic over all the samples you could have drawn. Its standard deviation is the **standard error** (SE). The **standard deviation** describes how much individual data points vary; the standard error describes how much a *computed number* varies between studies. Confusing them is the most common mistake here.

~~~dg-flow The sampling distribution is a thought experiment you can run in code: repeat the whole data collection, not just the calculation.
Population -(draw n)-> Sample -(compute)-> Statistic -(repeat many times)-> *Sampling distribution -(its std)-> Standard error
~~~

In real life you get one study, not 20,000, so we need the SE and the shape from that one study.

## Step 1: the standard error of a mean

Take $n$ independent draws $x_1, \dots, x_n$ from a population with mean $\mu$ and variance $\sigma^2$. The sample mean is $\bar x = \frac{1}{n}\sum_i x_i$. Two facts about variances do all the work:

- The variance of a sum of *independent* variables is the sum of their variances: $\mathrm{Var}(\sum_i x_i) = n\sigma^2$.
- Scaling a variable by $c$ scales its variance by $c^2$: dividing by $n$ divides the variance by $n^2$.

Put together:

$$\mathrm{Var}(\bar x) = \frac{n\sigma^2}{n^2} = \frac{\sigma^2}{n}, \qquad \mathrm{SE}(\bar x) = \frac{\sigma}{\sqrt n} \approx \frac{s}{\sqrt n}.$$

Here $\sigma$ is the population standard deviation (unknown in practice), $s$ the sample standard deviation (lesson 13.01, divide by $n-1$), and $n$ the number of independent observations. For the session example, $\sigma/\sqrt n = 6/\sqrt{25} = 1.2$, exactly the spread of the right-hand histogram. In study 1 we would estimate it as $5.67/5 = 1.13$.

<div class="ex">
<div class="lab">Worked example</div>

**a sampling distribution you can write out by hand**

Population: four customers who spent 2, 4, 6 and 8 euros. Its mean is $\mu = 5$. Deviations from the mean are $-3, -1, 1, 3$; their squares are $9, 1, 1, 9$, so the population variance is $\sigma^2 = 20/4 = 5$.

Draw a sample of $n = 2$ customers *with replacement* (so the two draws are independent). There are $4 \times 4 = 16$ equally likely ordered samples. List their averages: (2,2) gives 2; (2,4) and (4,2) give 3; (2,6), (4,4), (6,2) give 4; and so on.

Counting them: average 2 occurs once, 3 twice, 4 three times, 5 four times, 6 three times, 7 twice, 8 once. That table *is* the sampling distribution of $\bar x$.

Its mean: $(2 + 6 + 12 + 20 + 18 + 14 + 8)/16 = 80/16 = 5$, the population mean. Unbiased, as promised.

Its variance: the squared distances from 5 are $9, 4, 1, 0, 1, 4, 9$. Weighted by the counts: $1\cdot9 + 2\cdot4 + 3\cdot1 + 4\cdot0 + 3\cdot1 + 2\cdot4 + 1\cdot9 = 40$, and $40/16 = 2.5$.

The formula predicts $\sigma^2/n = 5/2 = 2.5$. They agree exactly, so $\mathrm{SE} = \sqrt{2.5} \approx 1.58$ euros, against a population standard deviation of $\sqrt 5 \approx 2.24$.

Notice the shape: the population was flat, but the averages form a triangle peaked in the middle. That is the CLT starting to work at $n = 2$.

</div>

## Step 2: the square-root law

Because $n$ sits under a square root, precision is expensive. To halve the standard error you need four times the data; to cut it by 10 you need 100 times.

~~~dg-fig 13.02-sqrtn.svg
The simulated standard error (dots) sits on the line 6/√n. On log-log axes that is a straight line with slope −1/2: every 4× increase in sample size halves the error.
~~~

The simulation in the figure confirms the formula to two decimals across the whole range: at $n = 100$ the spread of 4,000 study averages is 0.598 against a predicted 0.600; at $n = 400$ it is 0.299 against 0.300. That is why detecting an effect half as large needs four times as many users.

## Step 3: proportions are means too

A conversion rate is the mean of a column of 0s and 1s. A 0/1 variable with success rate $p$ has variance $p(1-p)$, so the same formula gives

$$\mathrm{SE}(\hat p) = \sqrt{\frac{p(1-p)}{n}},$$

where $\hat p$ is the observed fraction of successes and $n$ is the number of independent trials. The *relative* error $\mathrm{SE}/p$ grows as $p$ gets small, which is why rare events need huge samples.

<div class="ex">
<div class="lab">Worked example</div>

**how precise is a 2% click-through rate?**

You show an ad $n = 10{,}000$ times (independent impressions) and get 200 clicks, so $\hat p = 0.02$.

Variance of one impression: $0.02 \times 0.98 = 0.0196$. Divide by $n$: $0.0196/10{,}000 = 0.00000196$. Square root: $\mathrm{SE} = 0.0014$.

A 95% interval is $\hat p \pm 1.96\,\mathrm{SE} = 0.02 \pm 1.96 \times 0.0014 = 0.02 \pm 0.00274$, so roughly 1.73% to 2.27%.

In relative terms that is $0.00274/0.02 \approx 13.7\%$ either side of the estimate.

To reach $\pm 5\%$ relative precision, the error must shrink by a factor $13.7/5 = 2.74$, so the data must grow by $2.74^2 \approx 7.5$: about 75,000 impressions.

Check the normal approximation: $n\hat p = 200$ and $n(1-\hat p) = 9{,}800$, both far above 10, so it is fine. With the same $n$ and $p = 0.001$ you would expect only 10 clicks, and the bell curve would be shaky.

</div>

## Step 4: the central limit theorem

The SE tells you how wide the sampling distribution is. The **central limit theorem** tells you its shape. If $x_1, \dots, x_n$ are independent and identically distributed with mean $\mu$ and finite variance $\sigma^2$, then

$$\frac{\bar x - \mu}{\sigma/\sqrt n} \xrightarrow{d} N(0,1) \quad \text{as } n \to \infty.$$

In words: subtract the true mean, divide by the standard error, and the result looks more and more like a standard normal as $n$ grows, *whatever the shape of the data*. The $d$ on the arrow means "converges in distribution".

~~~dg-fig 13.02-clt.svg
Standardised means of exponential data. A single observation (n = 1) is sharply skewed; averaging 2, 5 and 30 observations pulls the histogram towards the dashed normal curve as the skewness falls like 2/√n.
~~~

The figure shows why it works. The exponential distribution has skewness 2. Averaging $n$ independent copies divides the skewness by $\sqrt n$: 1.41 at $n = 2$, 0.89 at $n = 5$, 0.37 at $n = 30$.

This is why you can say "the true mean is within $\bar x \pm 1.96\,\mathrm{SE}$ with about 95% confidence" without knowing what the population looks like. Lessons 13.03 and 13.06 build tests and A/B experiments on exactly this.

## Step 5: how large is "large enough"?

The theorem is about a limit; what matters is how fast you get there, and that depends on skew. The rule "$n \geq 30$" is fine for mild skew and badly wrong for heavy skew.

The honest check is to simulate the thing you will actually do: compute $\bar x \pm 1.96\, s/\sqrt n$ and count how often it misses the true mean. It is supposed to miss 5% of the time.

~~~dg-fig 13.02-rate.svg
Left: how often the everyday interval mean ± 1.96·s/√n misses the true mean. Symmetric data are on target by n = 30; exponential data come close by about n = 100; heavily skewed log-normal data still miss 7% of the time at n = 3000. Right: for the log-normal, nearly every miss is an interval sitting entirely below the true mean.
~~~

What the simulation shows:

- **Uniform (symmetric).** 6% misses at $n = 30$, 5% from $n = 100$. At $n = 5$ it misses 12.8% because 1.96 is too small when $s$ itself is noisy; the $t$ distribution (lesson 13.03) fixes that.
- **Exponential (skewness 2), like session lengths.** 8% misses at $n = 30$, 6% at $n = 100$, on target by a few hundred.
- **Log-normal with $\sigma = 1.5$ (skewness about 33), like revenue with a few big spenders.** 31% misses at $n = 10$, 15% at $n = 100$, still 7% at $n = 3{,}000$.

The right panel explains the log-normal failure. A small sample usually contains no big spender, so both $\bar x$ and $s$ come out too small together, and the interval sits confidently below the truth. When a whale does appear, $s$ explodes and the interval covers. Because the misses are one-sided, a test of "is revenue below target?" would be badly overconfident.

## Sampling distributions in code

The simulation habit is worth more than any rule of thumb: if you can write down a plausible population, you can measure the sampling distribution directly.

```python
import numpy as np
rng = np.random.default_rng(0)

# One real study: 25 sessions (minutes). Estimate the mean and its SE.
x = rng.exponential(6, size=25)
se = x.std(ddof=1) / np.sqrt(len(x))
print(round(x.mean(), 2), round(se, 2))        # 7.39 1.68

# The thought experiment, run for real: 10,000 studies of 25 sessions each.
means = rng.exponential(6, size=(10_000, 25)).mean(axis=1)
print(round(means.std(), 3), 6 / np.sqrt(25))  # 1.228 1.2

# Does mean +- 1.96*SE really cover the true mean 95% of the time?
def miss_rate(draw, true_mean, n, reps=10_000):
    s = draw((reps, n))
    se = s.std(axis=1, ddof=1) / np.sqrt(n)
    z = (s.mean(axis=1) - true_mean) / se
    return np.mean(np.abs(z) > 1.96)

lognorm = lambda size: rng.lognormal(0, 1.5, size)   # skewed "revenue"
for n in (10, 100, 1000):
    print(n, round(miss_rate(lognorm, np.exp(1.5**2 / 2), n), 3))
# 10 0.318
# 100 0.15
# 1000 0.084
```

Use `ddof=1` for $s$, put repeated studies in the rows of one array instead of a Python loop, and always compare against something you know (the theoretical 1.2, the promised 5%).

## Other statistics have standard errors too

The **median** of normal data has $\mathrm{SE} \approx 1.253\,\sigma/\sqrt n$, about 25% noisier than the mean: the price of ignoring how far out the extreme points are. On heavy-tailed data the order flips.

~~~dg-fig 13.02-median.svg
Simulated standard errors for samples of 50. On normal data the mean is more precise; on heavy-tailed data (Student t with 3 degrees of freedom, Laplace) the median wins, because single extreme values no longer drag it around.
~~~

In the simulation, on normal data the median's SE is 0.173 against the mean's 0.141 (a ratio of 1.23, close to the theoretical 1.253 which holds as $n$ grows). On Student-$t$ data with 3 degrees of freedom the median's SE is 0.191 against 0.238 for the mean, and on Laplace data 0.156 against 0.201. For statistics without a convenient formula (a p90 latency, a trimmed mean, an AUC, a ratio), the **bootstrap** in lesson 13.05 estimates the SE by resampling your one dataset.

## Reading and checking the result

Before you report an SE or an interval, run through these checks:

- **Is $n$ the number of independent units?** If you sampled users, $n$ counts users, not page views.
- **Is the data skewed?** If the largest few values carry a big share of the total, the normal approximation needs far more data; simulate or bootstrap.
- **For proportions, are the counts large?** You want at least about 10 successes and 10 failures; otherwise use an exact or Wilson interval (lesson 13.04).
- **Does a simulation agree?** Draw from a population like yours and check that the interval misses about 5% of the time.

## What goes wrong

When the assumptions fail, the error bars are wrong, usually too narrow.

~~~dg-fig 13.02-fail.svg
Left: running means of three Cauchy sequences keep jumping after thousands of observations; the normal running mean settles. Right: with lag-1 correlation ρ the true SE exceeds the naive s/√n by √((1+ρ)/(1−ρ)), three times at ρ = 0.8.
~~~

- **Infinite variance.** The CLT needs $\sigma^2 < \infty$. The mean of $n$ Cauchy draws has *exactly* the same distribution as a single draw, for every $n$: averaging does not help at all, as the left panel shows. Pareto tails with $1 < \alpha < 2$ have a mean but no variance; averages do converge, but slowly and to a non-normal stable law (at exactly $\alpha = 2$ the limit is normal again, but the error shrinks more slowly than $\sigma/\sqrt n$ suggests). Heavy-tailed revenue and file sizes come close to this regime.
- **Dependence.** For an AR(1) series with lag-1 correlation $\rho$, $\mathrm{Var}(\bar x) \approx \frac{\sigma^2}{n}\cdot\frac{1+\rho}{1-\rho}$. At $\rho = 0.5$ the variance triples, so the **effective sample size** is $n/3$; the simulation in the right panel matches the formula (SE ratio 1.73 at $\rho = 0.5$). Time series, repeated measures and clustered data all break the plain formula.
- **Statistics that are not averages.** The CLT says nothing about a maximum, a p99.9 or a distinct count; use extreme-value theory or a careful bootstrap.
- **Ratio metrics.** Clicks per page view, with users randomised, is a ratio of two correlated per-user means. Treating page views as independent understates the SE; use the delta method or a bootstrap over users.

| Situation | Symptom | Fix
| --- | --- | --- |
| Skewed data, small $n$ | Intervals miss far more than 5%, mostly on one side | Larger $n$, log transform, bootstrap
| Heavy tails (no variance) | Estimate jumps whenever a new extreme arrives | Median, trimming, winsorizing
| Correlated observations | SE too small, tests overconfident | Analyse per cluster or user, block bootstrap
| Ratio of means | Unit of analysis does not match randomisation | Delta method, aggregate per randomisation unit
| Rare events ($n\hat p < 10$) | Interval runs below 0 or is lopsided | Exact or Wilson interval, more data
| SD reported as SE | Error bars $\sqrt n$ times too wide (or vice versa) | Label clearly; SE $= s/\sqrt n$

## Going deeper

**Berry-Esseen: a speed limit on the CLT.** The theorem bounds the worst-case gap between the true CDF of the standardised mean, $F_n$, and the normal CDF $\Phi$:

$$\sup_z |F_n(z) - \Phi(z)| \le \frac{C\,\rho}{\sigma^3\sqrt n}, \qquad \rho = E|x-\mu|^3,$$

where $C$ is a universal constant below 0.5 (the best published value for identically distributed data is about 0.47, stated from memory). The bound shrinks like $1/\sqrt n$, and skewness enters through $\rho/\sigma^3$. For the exponential distribution $\rho/\sigma^3 \approx 2.41$, so at $n = 30$ the bound is about 0.21. The actual worst gap, computed exactly from the gamma distribution, is 0.024. The bound must hold for *every* distribution with that third moment, so it is very conservative; use it for the $1/\sqrt n$ rate and the role of skew, and simulation for actual numbers.

**Why skewness falls like $1/\sqrt n$.** Cumulants add over independent variables. The third cumulant of a sum is $n\kappa_3$ and its variance is $n\sigma^2$, so the skewness of the sum (and of the mean) is $n\kappa_3/(n\sigma^2)^{3/2} = \gamma/\sqrt n$, where $\gamma$ is the skewness of one observation. Excess kurtosis falls faster, like $1/n$, which is why skew is usually what slows the CLT in practice, and why the first Edgeworth correction to the normal approximation is a skewness term.

**Beyond identical distributions.** The Lindeberg and Lyapunov versions of the CLT allow the $x_i$ to differ in distribution, as long as no single term dominates the total variance: the formal version of "many small independent effects add up to a normal".

**The design effect.** For data in clusters of size $m$ with intra-cluster correlation $\rho$, the variance of the overall mean is inflated by about $1 + (m-1)\rho$. With $\rho = 0.05$ and $m = 41$ per cluster the variance already triples.

**Sampling without replacement** multiplies the variance by $(N-n)/(N-1)$ for a population of size $N$; negligible unless you sample a sizeable fraction of it.

## Key terms

| Term | Meaning
| --- | --- |
| Sampling distribution | The distribution of a statistic over all samples you could have drawn
| Standard error (SE) | The standard deviation of the sampling distribution; how much the statistic moves between studies
| $\sigma/\sqrt n$ | The SE of a mean of $n$ independent observations with standard deviation $\sigma$
| Central limit theorem | Standardised means of independent, finite-variance data approach a standard normal as $n$ grows
| Berry-Esseen bound | A worst-case limit on how far the standardised mean is from normal, shrinking like $1/\sqrt n$ and growing with skew
| Effective sample size | Number of independent observations giving the same SE as your correlated data
| Design effect | The factor by which clustering inflates the variance of a mean, about $1 + (m-1)\rho$
| Heavy tails | Extremes common enough that the variance is huge or infinite

<div class="take">
<div class="lab">Key takeaway</div>

the standard error, not the standard deviation, measures how much a statistic would move on fresh data; trust $\sigma/\sqrt n$ and the normal approximation only for independent, finite-variance data with enough observations to overcome the skew, and when unsure, simulate the whole study and count how often your interval misses.

</div>
