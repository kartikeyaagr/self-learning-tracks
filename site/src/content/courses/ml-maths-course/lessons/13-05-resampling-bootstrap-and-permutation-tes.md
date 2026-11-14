---
id: "13.05"
course: "ml-maths-course"
module: "13"
slug: "resampling-bootstrap-and-permutation-tests"
title: "Resampling: bootstrap and permutation tests"
keyIdeas: ["The **bootstrap** estimates a sampling distribution by resampling the data with replacement: the sample stands in for the population (the plug-in principle).", "Percentile intervals take the 2.5% and 97.5% quantiles of the bootstrap statistics; BCa intervals correct for bias and skew and are the better default when available.", "A **permutation test** builds the null distribution by shuffling group labels, which is exact when units are exchangeable under $H_0$; report $p = (1+\\#\\{T^b \\ge T\\})/(1+B)$.", "The bootstrap answers \"how uncertain is this estimate\", the permutation test answers \"could this difference be chance\"; both must resample the independent unit (users, not events).", "The bootstrap fails for extremes (the max), very small samples and dependent data unless you use a block or cluster bootstrap."]
question: "You bootstrap the p99 latency of 400 requests to get a 95% confidence interval, and the interval looks oddly narrow and lumpy, sitting on just a few distinct values. Explain why, and what you would do instead."
answer: "The p99 of 400 points is determined by the top 4 or 5 order statistics. Every bootstrap resample is built only from the observed values, so its p99 can only land on (or interpolate between) the few largest observations; the replicates take a handful of distinct values and cannot reach beyond the observed maximum. The bootstrap is known to be unreliable for extremes and high quantiles because the empirical distribution has no tail beyond the sample. Options: collect far more requests so the p99 rests on hundreds of tail points; report a lower, better-supported quantile such as p90; use a distribution-free order-statistic interval for the quantile, based on the binomial count of points below it; or fit a tail model (for example a generalized Pareto above a threshold) if extrapolation is needed. Also check that requests are independent; bursts of slow requests would call for a block bootstrap."
sources: [{"title": "SciPy: scipy.stats.bootstrap", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html", "note": "Percentile, basic and BCa intervals, paired and vectorised statistics."}, {"title": "SciPy: scipy.stats.permutation_test", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html", "note": "Independent, paired and correlation permutation types and the p-value convention."}, {"title": "DiCiccio and Efron: Bootstrap confidence intervals", "url": "https://projecteuclid.org/journals/statistical-science/volume-11/issue-3/Bootstrap-confidence-intervals/10.1214/ss/1032280214.full", "note": "Why BCa improves on percentile intervals, from the method's authors."}]
practice: ["bootstrap-confidence-interval", "random-array-generation", "canary-mann-whitney"]
tags: ["bootstrap", "permutation test"]
tools: ["NumPy", "SciPy"]
---

## When formulas run out

The tests in 13.03 and 13.04 rely on formulas for the standard error and on approximate normality. Many quantities you care about have no handy formula: the median latency, the AUC of a classifier, the ratio of two means, the 90th percentile, a difference in F1 scores. Resampling methods replace the algebra with computation. Both ideas below fit in ten lines of NumPy, and both are used in practice for model evaluation.

## The bootstrap

The ideal experiment for an SE (13.02) is to draw many fresh samples from the population. The bootstrap's trick, due to Efron (1979), is to draw them from the sample itself: treat the $n$ observed values as the population, draw $n$ values **with replacement**, recompute the statistic, and repeat $B$ times. The spread of the $B$ replicates estimates the sampling spread.

~~~dg-flow Bootstrap: resample with replacement, recompute, and read error bars off the replicates.
Sample of n -(draw n with replacement)-> Resample -(compute statistic)-> Replicate -(repeat B times)-> *Bootstrap distribution -(quantiles)-> Confidence interval
~~~

```
import numpy as np
rng = np.random.default_rng(0)
def bootstrap_ci(x, stat=np.median, B=10_000, level=0.95):
    idx = rng.integers(0, len(x), size=(B, len(x)))      # B resamples at once
    reps = stat(x[idx], axis=1)
    a = (1 - level) / 2
    return np.quantile(reps, [a, 1 - a]), reps.std(ddof=1)  # percentile CI, SE
```

Three intervals are common:

- **Percentile**: the $\alpha/2$ and $1-\alpha/2$ quantiles of the replicates. Simple, and respects the range of the statistic.
- **Basic (pivotal)**: reflects the percentile interval around the estimate, $[2\hat\theta - q_{1-\alpha/2},\ 2\hat\theta - q_{\alpha/2}]$.
- **BCa** (bias-corrected and accelerated): adjusts the percentiles for median bias and for how the SE changes with the parameter. It has better coverage on skewed statistics and is the default in `scipy.stats.bootstrap`.

$B = 1{,}000$ is enough for an SE; for 95% interval endpoints use 5,000 to 10,000, because tail quantiles are noisy.

Where it fails:

- **Extremes.** The bootstrap distribution of the sample maximum puts about 63% of its mass on the observed maximum, because a resample contains the largest value with probability $1-(1-1/n)^n \approx 1 - 1/e$. High quantiles such as p99 from a few hundred points have the same problem.
- **Tiny samples.** With $n = 8$ the sample is a poor stand-in for the population and intervals are too narrow.
- **Dependence.** Resampling individual events from correlated data destroys the correlation and understates the SE. Resample whole clusters (users, sessions) or contiguous blocks of a time series.

## Permutation tests

A permutation test answers a different question: is the observed difference between groups larger than what random labelling would produce? Under $H_0$ the group labels are arbitrary (the units are **exchangeable**), so every reassignment of labels was equally likely to have occurred.

1. Compute the observed statistic $T$, for example the difference in means.
2. Shuffle the labels, recompute the statistic $T^b$; repeat $B$ times (or enumerate all reassignments when that is feasible).
3. $p = \dfrac{1 + \#\{b : T^b \ge T\}}{1 + B}$ for a one-sided test, using $|T|$ for two-sided. The $+1$ counts the observed labelling as one of the permutations, which keeps $p$ valid and never exactly zero.

The test makes no normality assumption and works with any statistic: a difference in medians, AUC or a custom business metric. Its assumption is exchangeability. If the groups have very different variances, a permutation test of means can reject because of the variance difference alone; if units are clustered, permute cluster labels, not rows.

<div class="ex">
<div class="lab">Worked example</div>

**four latency measurements per configuration**

Old: 3, 5, 4, 6 (mean 4.5). New: 7, 8, 6, 9 (mean 7.5). Observed difference $T = 3.0$.

There are $\binom{8}{4} = 70$ ways to split the eight values into two groups of four. Exactly 2 give a difference of at least 3.0 (the observed split, and the one that swaps the two 6s).

Exact one-sided $p = 2/70 = 0.029$; two-sided $p = 4/70 = 0.057$. With so few points, 0.029 is close to the smallest p-value achievable ($1/70 = 0.014$), a useful reminder of how little 8 numbers can prove.

</div>

## Which one to use

| Goal | Method | Resamples
| --- | --- | --- |
| Error bars on an estimate | bootstrap | with replacement, within the sample
| Test "no difference between groups" | permutation test | shuffle labels, without replacement
| CI for a difference between two groups | bootstrap each group separately | with replacement, per group
| Data clustered by user | either, at user level | whole users

The two are complementary. For evaluating models, a paired bootstrap over test examples (resample example indices, recompute both models' metrics on the same resample) gives an interval for the metric difference; a paired permutation test (randomly swap the two models' predictions per example) tests whether that difference is zero. The bootstrap also appears in the Machine Learning course, where bagging and random forests average models fitted on bootstrap resamples.

<div class="take">
<div class="lab">Key takeaway</div>

bootstrap the independent unit to get error bars for any statistic, permute labels to test whether a difference beats chance, and do not trust either for sample maxima, tiny samples or data whose dependence the resampling ignores.

</div>
