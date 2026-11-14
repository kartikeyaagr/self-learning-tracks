---
id: "13.05"
course: "ml-maths-course"
module: "13"
slug: "resampling-bootstrap-and-permutation-tests"
title: "Resampling: bootstrap and permutation tests"
keyIdeas: ["The **bootstrap** treats the sample as the population (the plug-in principle): draw $n$ values with replacement, recompute the statistic, repeat thousands of times, and the spread of these replicates estimates the standard error.", "Percentile intervals take the 2.5% and 97.5% quantiles of the replicates; BCa intervals correct for bias and skew and are the better default (SciPy's `stats.bootstrap` uses it). Use 5,000 to 10,000 resamples for interval endpoints.", "A **permutation test** builds the null distribution by shuffling group labels, which is exact when units are exchangeable under $H_0$; report $p = (1+\\#\\{T^b \\ge T\\})/(1+B)$.", "The bootstrap answers \"how uncertain is this estimate\", the permutation test answers \"could this difference be chance\". When two models share a test set, resample the shared examples (paired), which gives much tighter intervals.", "Both must resample the independent unit (users, not events; blocks of a time series, not points). The bootstrap also fails for extremes such as the maximum or p99 and is unreliable on very small samples."]
question: "You bootstrap the p99 latency of 400 requests to get a 95% confidence interval, and the interval looks oddly narrow and lumpy, sitting on just a few distinct values. Explain why, and what you would do instead."
answer: "The p99 of 400 points is determined by the top 4 or 5 order statistics. Every bootstrap resample is built only from the observed values, so its p99 can only land on (or interpolate between) the few largest observations; the replicates take a handful of distinct values and cannot reach beyond the observed maximum. This is the same failure as the bootstrap of the sample maximum, where about 63% of resamples ($1-(1-1/n)^n \\approx 1-1/e$) contain the observed maximum: the empirical distribution has no tail beyond the sample, so the bootstrap cannot represent uncertainty out there and the interval is too narrow. Options: collect far more requests so the p99 rests on hundreds of tail points; report a lower, better-supported quantile such as p90; use a distribution-free order-statistic interval for the quantile, based on the binomial count of points below it (the number of observations below the true p99 is Binomial(400, 0.99), so choose order statistics whose ranks cover 95% of that distribution); or fit a tail model (for example a generalized Pareto above a threshold) if extrapolation is needed. Also check that requests are independent; bursts of slow requests would call for a block bootstrap, and if requests come from a few heavy users, resample users rather than requests."
sources: [{"title": "SciPy: scipy.stats.bootstrap", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html", "note": "Percentile, basic and BCa intervals, paired and vectorised statistics."}, {"title": "SciPy: scipy.stats.permutation_test", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html", "note": "Independent, paired and correlation permutation types and the p-value convention."}, {"title": "DiCiccio and Efron: Bootstrap confidence intervals", "url": "https://projecteuclid.org/journals/statistical-science/volume-11/issue-3/Bootstrap-confidence-intervals/10.1214/ss/1032280214.full", "note": "Why BCa improves on percentile intervals, from the method's authors."}]
practice: ["bootstrap-confidence-interval", "random-array-generation", "canary-mann-whitney"]
tags: ["bootstrap", "permutation test"]
tools: ["NumPy", "SciPy"]
---

## What you will learn

- Why resampling exists: many statistics (a median, an AUC, a p90) have no standard-error formula.
- How the **bootstrap** turns one sample into thousands of "pretend" samples to get error bars.
- How to read three bootstrap intervals (percentile, basic, BCa) and how many resamples you need.
- How a **permutation test** builds a null distribution by shuffling labels, and how to compute its p-value correctly.
- The paired versions used to compare two ML models on the same test set.
- Where both methods break: sample maxima, tiny samples and dependent data.

*Before you start:* you need the idea of a standard error and a sampling distribution (13.02), and of a null hypothesis and a p-value (13.03). Quantiles from 13.01 appear throughout. The code uses NumPy and SciPy.

## When formulas run out

Lessons 13.03 and 13.04 tested means, counts and correlations, each with a standard-error (SE) formula and a known null distribution ($t$, $\chi^2$, $F$). That stops working the moment you ask about the median latency, a classifier's AUC, the 90th percentile of revenue, the ratio of two means or a difference in F1 scores.

Resampling replaces the algebra with a loop. The **bootstrap** asks "how much would this number wobble if I collected the data again?" The **permutation test** asks "could random labelling alone produce this difference?" Both fit in ten lines of NumPy.

## The idea in one picture

A service logs 20 requests. Their latencies in milliseconds, sorted:

| Requests 1-10 | 26 | 36 | 64 | 71 | 80 | 85 | 91 | 95 | 97 | 101
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Requests 11-20 | 104 | 104 | 105 | 106 | 137 | 154 | 160 | 213 | 408 | 881

The data are skewed: the mean is 155.9 ms, pulled up by the 881 ms request, while the median is 102.5 ms. You report the median. How precise is it? There is no simple formula for the SE of a median from 20 skewed values.

~~~dg-fig 13.05-idea.svg
Left: the 20 latencies; the median sits in the main cluster, the mean is dragged right by two slow requests. Right: medians of 10 000 resamples. Their spread is the bootstrap's estimate of how much the median would vary between samples; the middle 95% gives an interval of 88.0 to 121.5 ms.
~~~

The recipe behind the right panel:

1. Draw 20 values from the list **with replacement** (some values appear twice, some not at all). This is one **resample**.
2. Compute the median of that resample. This is one **replicate**.
3. Repeat 10 000 times and look at the histogram of replicates.

The replicates have a standard deviation of about 8.3 ms (the **bootstrap SE**), and their 2.5% and 97.5% quantiles give a 95% **percentile interval** of 88.0 to 121.5 ms. Notice the histogram is spiky: a median of 20 numbers can only be one of the data values or the average of two neighbours, so only 57 distinct values appeared. That is normal for medians and a hint of what goes wrong at the extremes (see What goes wrong).

## Step 1: the plug-in principle

The ideal way to get an SE (13.02) is to draw many fresh samples from the population and measure the statistic's spread. You have one sample. The bootstrap (Efron, 1979) makes one bold substitution: the sample is your best picture of the population, so **treat the sample as if it were the population** and draw fresh samples from it. In symbols, the population distribution $F$ is replaced by the **empirical distribution** $\hat F_n$, which puts probability $1/n$ on each observed value. Any quantity defined in terms of $F$ is estimated by computing it under $\hat F_n$: this is the **plug-in principle**.

| Real world | Bootstrap world
| --- | --- |
| Population $F$ (unknown) | Empirical distribution $\hat F_n$ (the sample)
| Sample of size $n$ from $F$ | Resample of size $n$ from $\hat F_n$, with replacement
| Estimate $\hat\theta$ | Replicate $\hat\theta^*$
| Spread of $\hat\theta$ around $\theta$ | Spread of $\hat\theta^*$ around $\hat\theta$

The key bet is the last row: replicates scatter around your estimate the way your estimate scatters around the truth. The bootstrap does not fix a wrong estimate; it only measures its wobble.

~~~dg-fig 13.05-plugin.svg
Grey: the true sampling distribution of the median of a log-normal population (median 120 ms). Coloured: four bootstrap distributions, each from a single sample. At n = 20 their widths range from SE 12.7 to 44.2 against a true 19.9; at n = 200 they range from 4.9 to 6.8 against a true 6.4.
~~~

Each bootstrap curve is centred on its own sample's median, not on 120; only its width is used. At $n = 200$ every width is close to the truth. At $n = 20$ the widths vary by a factor of three: a rough picture of the population gives a rough picture of the sampling distribution.

## Step 2: resampling with replacement, by hand

Why *with* replacement? Without it, a resample is just the sample reshuffled and every replicate equals $\hat\theta$. Replacement creates the variation, mimicking independent draws.

A typical resample contains about 63% of the distinct original values. A given value is missed by one draw with probability $1 - 1/n$, so it is missing from the whole resample with probability $(1-1/n)^n \approx e^{-1} = 0.368$.

<div class="ex">
<div class="lab">Worked example</div>

**the complete bootstrap of a mean, n = 3**

Data: $x = (1, 2, 6)$, mean $\bar x = 3$. With $n = 3$ there are $3^3 = 27$ equally likely ordered resamples, so we can list the whole bootstrap distribution instead of sampling it.

A few of them: $(1,1,1) \to 1$; $(1,2,6) \to 3$; $(6,6,2) \to 4.67$; $(6,6,6) \to 6$. Six orderings of $(1,2,6)$ give 3, three orderings of $(1,1,2)$ give 1.33, and so on.

The mean of all 27 replicates is exactly 3. The bootstrap is centred on the estimate.

Their variance can be found without listing them: each draw has the plug-in variance $\hat\sigma^2 = \frac{1}{3}\big((1-3)^2 + (2-3)^2 + (6-3)^2\big) = \frac{4 + 1 + 9}{3} = \frac{14}{3}$, and a mean of 3 independent draws has variance $\frac{14/3}{3} = \frac{14}{9} = 1.556$.

Bootstrap SE: $\sqrt{14/9} = 1.247$. The usual formula $s/\sqrt n$ uses $s^2 = 14/2 = 7$ and gives $\sqrt{7/3} = 1.528$.

The ratio is $1.247 / 1.528 = 0.816 = \sqrt{2/3} = \sqrt{(n-1)/n}$. The bootstrap uses the divide-by-$n$ variance (13.01), so it slightly underestimates spread in small samples; at $n = 100$ the factor is 0.995 and nobody cares.

</div>

Real samples are too large to enumerate ($n^n$ ordered resamples, $\binom{2n-1}{n}$ distinct ones: 10 for $n = 3$), so you draw $B$ of them at random.

## Step 3: from replicates to an interval

Call the replicates $\hat\theta^*_1, \dots, \hat\theta^*_B$ and let $q_p$ be their $p$-quantile. With confidence level $1 - \alpha$ (for 95%, $\alpha = 0.05$), three intervals are standard:

- **Percentile**: $[q_{\alpha/2},\ q_{1-\alpha/2}]$. Read the middle 95% of the histogram directly. Simple, and it never leaves the allowed range of the statistic (an accuracy interval stays inside 0 to 1).
- **Basic** (or pivotal): $[2\hat\theta - q_{1-\alpha/2},\ 2\hat\theta - q_{\alpha/2}]$. It assumes $\hat\theta - \theta$ behaves like $\hat\theta^* - \hat\theta$, which flips the percentile interval around the estimate.
- **BCa** (bias-corrected and accelerated): still uses quantiles of the replicates, but shifts which quantiles it takes. A **bias correction** $z_0$ measures how far the median of the replicates sits from $\hat\theta$; an **acceleration** $a$, estimated by the jackknife (leaving one point out at a time), measures how fast the SE changes with the parameter. It is the default in `scipy.stats.bootstrap`.

For the latency median, SciPy's BCa interval (seeded, 9 999 resamples) is 88.0 to 137.0 ms, while its percentile interval is 88.0 to 129.0. BCa stretches the upper end because the replicates are skewed right.

The honest test of an interval is **coverage**: the share of intervals that contain the truth, measured by simulation.

~~~dg-fig 13.05-coverage.svg
Coverage of nominal 95% intervals for the mean of a log-normal population, 1000 simulated samples each. At n = 20 every method falls short (88.1% to 92.2%); at n = 80 all are within two points of 95%.
~~~

The bootstrap is not magic at small $n$: for a plain mean at $n = 20$ the $t$ interval did slightly better, partly because of the $\sqrt{(n-1)/n}$ narrowness from Step 2. Its real value is for statistics with no $t$ interval at all.

**How many resamples?** $B$ controls only Monte Carlo noise. $B = 1{,}000$ is plenty for an SE; tail endpoints need $B = 5{,}000$ to $10{,}000$. If endpoints move with the seed in a digit you would report, raise $B$.

## Step 4: the permutation test

A permutation test answers a yes/no question: is the difference between two groups bigger than chance labelling would produce? Under the null hypothesis $H_0$ "the configuration makes no difference", each measurement would have come out the same whichever group it was in. The group labels are then arbitrary, and every way of assigning them was equally likely. Units with this property are called **exchangeable**. So:

1. Compute the observed statistic $T$, for example mean(new) minus mean(old).
2. Shuffle the labels (without replacement, keeping group sizes), recompute the statistic $T^b$. Repeat $B$ times, or list every possible relabelling when there are few enough.
3. The p-value is the share of relabellings at least as extreme as the observed one:

$$p = \frac{1 + \#\{b : T^b \ge T\}}{1 + B}.$$

Here $\#\{\cdot\}$ counts shuffles. For a two-sided test compare $|T^b|$ with $|T|$. The $+1$ in numerator and denominator counts the observed labelling as one of the possibilities. It keeps the test valid (the false-positive rate never exceeds $\alpha$) and means $p$ can never be exactly zero, which no finite experiment can justify.

<div class="ex">
<div class="lab">Worked example</div>

**four latency measurements per configuration**

Old configuration: 3, 5, 4, 6 ms (mean 4.5). New: 7, 8, 6, 9 ms (mean 7.5). Observed difference $T = 7.5 - 4.5 = 3.0$.

Pool the eight values: 3, 4, 5, 6, 6, 7, 8, 9 (sum 48). There are $\binom{8}{4} = \frac{8!}{4!\,4!} = 70$ ways to choose which four are labelled "new".

If the "new" group sums to $S$, the other sums to $48 - S$, so the difference is $\frac{S}{4} - \frac{48 - S}{4} = \frac{2S - 48}{4}$. A difference of at least 3 needs $2S - 48 \ge 12$, so $S \ge 30$.

The largest possible sum is $9 + 8 + 7 + 6 = 30$, and there are two 6s, so exactly 2 splits reach 30: the observed one and the one using the other 6.

One-sided exact $p = 2/70 = 0.029$. By symmetry 2 splits have $S = 18$ (difference $-3$), so two-sided $p = 4/70 = 0.057$.

With all splits listed no $+1$ is needed: the observed split is already among the 70. The smallest possible p-value is $1/70 = 0.014$; eight numbers cannot prove much.

</div>

~~~dg-fig 13.05-perm.svg
The difference in means for all 70 ways to relabel the eight latencies. Under the null every bar is equally likely; only the 2 + 2 orange splits are as extreme as the observed 3.0, giving two-sided p = 4/70 = 0.057.
~~~

The test needs no normality and no SE formula, and works with any statistic (medians, AUC, a business metric). Rank-based tests such as Mann-Whitney (13.04) are permutation tests applied to ranks, which is why their null distributions could be tabulated in advance.

## Step 5: comparing two models on one test set

The most common use in ML: models A and B are scored on the same 500 test examples. The examples are shared, so the comparison is **paired**, and the resampling must respect that.

- **Paired bootstrap**: resample example indices, and compute both models' metrics on the same resample. The replicate is the difference.
- **Paired permutation test**: under $H_0$ "the models are equally good", which model produced which prediction on an example is arbitrary, so for each example swap A and B with probability one half.

Pairing matters because most examples are easy or hard for both models, so per-example noise cancels. (For two *independent* groups, bootstrap each group separately instead.) In the run below, the paired 95% interval for the accuracy gain is 1.2 to 6.2 points, while resampling the two models independently gives $-1.0$ to $8.0$ points, which wrongly includes zero.

## Resampling in code

```python
import numpy as np
from scipy import stats

x = np.array([26, 36, 64, 71, 80, 85, 91, 95, 97, 101, 104, 104, 105, 106, 137, 154, 160, 213, 408, 881.])

# 1. Bootstrap by hand: B resamples at once, as a (B, n) index matrix
rng = np.random.default_rng(0)
idx = rng.integers(0, len(x), size=(10_000, len(x)))
reps = np.median(x[idx], axis=1)
print(np.quantile(reps, [0.025, 0.975]), reps.std(ddof=1))    # [ 88.  121.5] 8.26

# 2. SciPy: BCa by default
res = stats.bootstrap((x,), np.median, n_resamples=9999, rng=1)
print(res.confidence_interval)                                 # low=88.0, high=137.0

# 3. Exact permutation test (SciPy enumerates all 70 splits by itself)
old, new = np.array([3, 5, 4, 6.]), np.array([7, 8, 6, 9.])
diff = lambda a, b, axis: b.mean(axis=axis) - a.mean(axis=axis)
pt = stats.permutation_test((old, new), diff, alternative='greater', vectorized=True)
print(pt.pvalue, len(pt.null_distribution))                    # 0.0286 70

# 4. Paired bootstrap for an accuracy difference (ca, cb: boolean "correct" per example)
rng = np.random.default_rng(3); n = 500
ca = rng.random(n) < 0.80
cb = ca.copy()
cb[rng.choice(np.flatnonzero(~ca), 30, replace=False)] = True   # B fixes 30 of A's errors
cb[rng.choice(np.flatnonzero(ca), 12, replace=False)] = False   # and breaks 12 successes
ii = rng.integers(0, n, (10_000, n))                            # SAME indices for both models
d = cb[ii].mean(1) - ca[ii].mean(1)
print(cb.mean() - ca.mean(), np.quantile(d, [0.025, 0.975]))   # 0.036 [0.012 0.062]
```

Indexing with a $(B, n)$ matrix replaces a Python loop. `stats.permutation_test` switches to exact enumeration when the number of distinct relabellings is no more than `n_resamples` (9 999 by default) and otherwise adds the $+1$ correction itself. For the two-sided version of call 3 it returns 0.057, matching the hand count.

## Reading and checking the result

- **Plot the replicates.** A smooth, roughly bell-shaped histogram is reassuring. Spikes, gaps or a large pile on one value (as with the maximum below) mean the statistic is too discrete for the bootstrap at this $n$.
- **Compare with a formula when one exists.** For the latency *mean*, the bootstrap SE is 40.7 ms and $s/\sqrt n$ gives 42.2 ms, about the $\sqrt{19/20}$ ratio from Step 2. A large disagreement points to a bug, such as resampling the wrong axis.
- **Compare interval methods.** If percentile and BCa disagree a lot (here 121.5 or 129 versus 137 at the top), the distribution is skewed or lumpy and you should report the uncertainty with caution or collect more data.
- **Check the unit.** Resample whatever was independently sampled or randomised: users, not their page views; documents, not their sentences.

## What goes wrong

Resampling makes few assumptions, but not none.

**Extremes.** A resample contains the sample maximum with probability $1 - (1 - 1/n)^n \approx 1 - 1/e = 0.632$. So about 63% of bootstrap maxima equal the observed maximum, and none can exceed it.

~~~dg-fig 13.05-max.svg
Fifty uniform values. Left: the true distribution of the maximum over fresh samples is smooth. Right: in 10 000 resamples, 64% have exactly the observed maximum; the bootstrap distribution is a few spikes and can never go higher.
~~~

High quantiles behave the same way: the p99 of 400 requests rests on the top four or five values.

**Dependent data.** Resampling single points assumes independence. In a time series neighbours are alike, so the data carry less information than $n$ suggests.

~~~dg-fig 13.05-block.svg
An AR(1) series of 200 points with correlation 0.7 between neighbours. The naive bootstrap SE of the mean is 0.10, less than half the true 0.23. Resampling blocks of 10 to 20 consecutive points gives 0.20; blocks of 40 leave too few blocks and become noisy.
~~~

Resample the truly independent units: whole users (a **cluster bootstrap**) or contiguous chunks (a **block bootstrap**). Likewise, permute user labels, not rows, when users were randomised.

| Problem | Symptom | Fix
| --- | --- | --- |
| Statistic is an extreme (max, p99) | spiky replicates, CI stuck at the observed max | more data, lower quantile, order-statistic interval
| Tiny sample ($n$ below about 15) | intervals too narrow, vary a lot between samples | report with caution, exact methods, collect more
| Events within users | SE suspiciously small | cluster bootstrap over users
| Time series | SE far below the block-bootstrap SE | moving-block bootstrap, blocks of several autocorrelation lengths
| Unequal variances in a permutation test of means | rejects too often with no mean difference | studentised statistic (Welch $t$) or bootstrap the difference
| Too few resamples | endpoints change with the seed | $B \ge 5{,}000$ for intervals
| Unpaired resampling of paired data | interval far too wide | resample the shared index for both models

## Going deeper

**Why it works.** For smooth statistics (means, smooth functions of means, most M-estimators), the bootstrap distribution of $\sqrt n(\hat\theta^* - \hat\theta)$ converges to the same limit as $\sqrt n(\hat\theta - \theta)$, so bootstrap intervals have asymptotically correct coverage. Edgeworth expansions refine this: the percentile and basic intervals have coverage error of order $1/\sqrt n$ for one-sided limits, while BCa and the **bootstrap-t** (resampling the studentised statistic $(\hat\theta^* - \hat\theta)/\widehat{SE}^*$) achieve order $1/n$. This "second-order accuracy" is why BCa is the recommended default.

**When it fails, formally.** The bootstrap is inconsistent for statistics that depend on the edge of the support (the maximum) or are not smooth at the true parameter ($|\bar x|$ when the mean is zero); the **m-out-of-n bootstrap** (resamples of size $m \ll n$) repairs many of these.

**Exactness of permutation tests.** Under exchangeability the observed labelling is a uniform draw from all relabellings, so $P(p \le \alpha) \le \alpha$ for any $n$ and statistic; the $+1$ keeps this under Monte Carlo sampling. The catch is the null: exchangeability means the *entire distributions* are equal, so a permutation test of means can reject because variances or shapes differ. Studentising the statistic restores approximate validity for the weaker "equal means" null.

**Beyond inference.** Bagging and random forests average models fit on bootstrap resamples; the roughly 37% of points left out of each resample give the **out-of-bag** error estimate, the same $e^{-1}$ from Step 2.

## Key terms

| Term | Meaning
| --- | --- |
| Bootstrap | Estimating a sampling distribution by resampling the data with replacement
| Plug-in principle | Replace the unknown population with the empirical distribution of the sample
| Replicate | The statistic computed on one resample
| Bootstrap SE | Standard deviation of the replicates
| Percentile / basic / BCa interval | Three ways to turn replicates into a confidence interval; BCa corrects for bias and skew
| Permutation test | A test whose null distribution comes from shuffling group labels
| Exchangeable | Labels can be swapped without changing the joint distribution, as under $H_0$
| Paired resampling | Resampling shared units (test examples) together so per-unit noise cancels
| Block / cluster bootstrap | Resampling chunks of a series or whole users so dependence is preserved

<div class="take">
<div class="lab">Key takeaway</div>

bootstrap the independent unit to get error bars for any statistic, permute labels to test whether a difference beats chance, pair the resampling when two models share a test set, and do not trust either method for sample maxima, tiny samples or data whose dependence the resampling ignores.

</div>
