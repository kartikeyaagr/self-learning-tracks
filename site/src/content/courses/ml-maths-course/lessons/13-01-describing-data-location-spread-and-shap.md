---
id: "13.01"
course: "ml-maths-course"
module: "13"
slug: "describing-data-location-spread-and-shape"
title: "Describing data: location, spread and shape"
keyIdeas: ["Location answers \"where is the typical value\": the **mean** uses every value, the **median** only the order, so one bad value can move the mean anywhere but cannot move the median past the middle.", "Spread comes in a fragile form (variance and standard deviation, with the $n-1$ divisor for samples) and robust forms (IQR, and MAD scaled by 1.4826 to match $\\sigma$ for normal data).", "Quantiles need an interpolation rule; NumPy's default `linear` is Hyndman-Fan type 7. Small samples give visibly different answers under different rules.", "Skewness and excess kurtosis summarise shape; latency, revenue and counts are usually right-skewed and heavy-tailed, which is exactly when means and 3-sigma rules mislead.", "A z-score outlier rule can hide the outlier it is looking for (masking), because the outlier inflates the standard deviation. Use median and IQR or MAD fences instead."]
question: "A teammate flags outliers in a column of per-user purchase amounts by removing rows with |z| > 3, recomputes the mean and standard deviation, and repeats until nothing is removed. What can go wrong, and what would you do instead?"
answer: "Purchase amounts are right-skewed with a heavy tail, so the first pass is distorted by masking: the biggest values inflate the standard deviation and can hide each other, so some true outliers survive. Iterating then has the opposite failure: each pass shrinks $s$, so ordinary large but legitimate purchases start crossing the threshold and the procedure keeps eating the right tail, biasing the mean and total revenue downwards. Instead, compute fences from order statistics (Tukey's $Q_3 + 1.5\\,\\mathrm{IQR}$ or a MAD-based robust z with a cutoff near 3.5), which the outliers cannot distort, then inspect what is flagged. Remove only values shown to be errors; for genuine whales, report a robust summary (median, trimmed mean) alongside the mean, or winsorize for modelling, rather than deleting revenue that really happened."
sources: [{"title": "NIST/SEMATECH e-Handbook: Measures of Skewness and Kurtosis", "url": "https://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm", "note": "Definitions and the effect of heavy tails on these moments."}, {"title": "NIST/SEMATECH e-Handbook: Detection of Outliers", "url": "https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm", "note": "Why masking happens and the modified (robust) z-score rule."}, {"title": "Hyndman and Fan: Sample quantiles in statistical packages", "url": "https://robjhyndman.com/publications/quantiles/", "note": "The nine quantile definitions that NumPy and R implement."}, {"title": "NumPy: numpy.quantile", "url": "https://numpy.org/doc/stable/reference/generated/numpy.quantile.html", "note": "The method argument and which Hyndman-Fan type each one is."}]
practice: ["percentiles-quantiles", "skewness-and-kurtosis", "mean-median-mode", "sample-variance-and-standard-deviation", "winsorize"]
tags: ["robust statistics", "quantiles"]
tools: ["NumPy", "SciPy"]
---

## Why start with description

This module picks up where the Inference module left off. Modules 10 to 12 treated random variables and estimators in theory; here the subject is the numbers you actually hold: a column of latencies, a list of per-user revenues, a vector of model errors. Every later lesson (standard errors, tests, A/B experiments) is built on a few summary statistics, so it is worth knowing exactly how each one reacts to bad data.

The three questions to ask of any sample are: where is it (location), how wide is it (spread) and what shape is it (skew, tails, outliers). Each has a fragile answer that uses every value with equal weight and a robust answer that only uses order.

## Location

- **Mean** $\bar x = \frac1n\sum_i x_i$. Optimal for symmetric light-tailed data and the only location statistic that adds up: total revenue equals mean revenue times users. One value of $10^9$ drags it anywhere.
- **Median**: the middle order statistic (average of the two middle ones when $n$ is even). It minimises $\sum_i |x_i - m|$, where the mean minimises squared error.
- **Trimmed mean**: drop the top and bottom $k\%$ and average the rest. A 10% trimmed mean is a compromise that keeps most of the mean's efficiency.
- **Mode**: the most frequent value; useful for categorical or heavily discretised data, close to useless for continuous measurements.

The **breakdown point** measures robustness: the fraction of values you can replace with garbage before the statistic can be made arbitrarily large. The mean's is $0$ (one value is enough), a 10% trimmed mean's is 10%, the median's is 50%.

## Spread

The sample variance is $s^2 = \frac{1}{n-1}\sum_i (x_i - \bar x)^2$. The $n-1$ (Bessel's correction) makes $s^2$ unbiased for $\sigma^2$, because deviations are measured from $\bar x$, which is itself fitted to the data and sits closer to the points than the true mean does. In NumPy this is `ddof=1`; the default `ddof=0` divides by $n$, a classic silent discrepancy with pandas, whose default is `ddof=1`.

Robust alternatives:

- **IQR** $= Q_3 - Q_1$, the width of the middle half.
- **MAD** $= \mathrm{median}_i\,|x_i - \mathrm{median}(x)|$. For normal data $\sigma \approx 1.4826\cdot\mathrm{MAD}$ and $\sigma \approx \mathrm{IQR}/1.349$, so both can stand in for a standard deviation.

| Statistic | Measures | Breakdown point | Use when
| --- | --- | --- | --- |
| Mean | location | 0% | data are clean and roughly symmetric, or totals matter
| Median | location | 50% | skewed data, outliers, latency
| Std deviation | spread | 0% | feeding normal-theory formulas
| IQR | spread | 25% | box plots, outlier fences
| MAD | spread | 50% | robust z-scores

## Quantiles are a choice, not a fact

The $p$-quantile is a value with a fraction $p$ of the data below it, but with $n$ points most $p$ fall between two order statistics. NumPy's `np.quantile(x, p)` defaults to `method="linear"`: sort, compute position $h = (n-1)p$, and interpolate between the values at $\lfloor h\rfloor$ and $\lceil h\rceil$. This is type 7 of the nine rules catalogued by Hyndman and Fan; R uses the same default. With 10 values the 90th percentile under different rules can differ by a whole order-statistic gap, which matters when you report p99 latency from a few hundred requests. State the rule, and prefer more data over a cleverer rule.

## Shape: skewness and kurtosis

With central moments $m_k = \frac1n\sum_i (x_i-\bar x)^k$:

$$g_1 = \frac{m_3}{m_2^{3/2}}, \qquad g_2 = \frac{m_4}{m_2^{2}} - 3$$

$g_1 > 0$ means a long right tail (mean above median), and $g_2$ is excess kurtosis, $0$ for a normal. Both are themselves very sensitive to outliers because they raise deviations to the third and fourth power; treat them as flags, not precise measurements. In SciPy, `stats.skew` and `stats.kurtosis` default to these biased, Fisher (excess) forms.

## Outliers and the masking trap

Two common outlier rules:

1. **z-score**: flag $|x_i - \bar x|/s > 3$.
2. **Tukey fences**: flag values below $Q_1 - 1.5\,\mathrm{IQR}$ or above $Q_3 + 1.5\,\mathrm{IQR}$. A **robust z-score** $(x_i - \mathrm{median})/(1.4826\cdot\mathrm{MAD})$ with a cutoff near 3.5 is the same idea on a different scale.

The z-score rule fails exactly when you need it: a large outlier inflates both $\bar x$ and $s$, so its own z-score shrinks. With several outliers they can hide each other completely.

~~~dg-flow Robust outlier screening: every step uses order statistics, so the outlier cannot distort its own yardstick.
Raw sample -(sort)-> Q1, median, Q3 -(IQR or MAD)-> Robust scale -(fences)-> *Flag outliers -(inspect)-> Fix, keep or model
~~~

<div class="ex">
<div class="lab">Worked example</div>

**ten request latencies in ms: 12, 13, 13, 14, 15, 15, 16, 18, 22, 480**

Location: mean $= 61.8$, median $= 15$. The single 480 ms request quadruples the mean.

Spread: $s = 147.0$, while MAD $= 2$ (robust scale $\approx 2.97$) and IQR $= 17.5 - 13.25 = 4.25$ with NumPy's default quantiles.

z-score of 480: $(480 - 61.8)/147.0 = 2.85$, below 3, so the 3-sigma rule does not flag it.

Robust z-score of 480: $(480 - 15)/2.97 \approx 157$. Tukey's upper fence is $17.5 + 1.5\cdot 4.25 = 23.9$, which flags 480 and keeps 22.

</div>

```
import numpy as np
x = np.array([12, 13, 13, 14, 15, 15, 16, 18, 22, 480.0])
q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75])
iqr = q3 - q1
mad = np.median(np.abs(x - med))
robust_z = (x - med) / (1.4826 * mad)
flag = (x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)
```

Flagging is not deleting. An outlier may be a logging bug (drop it), a real but rare event (keep it, and maybe report a tail percentile separately), or the signal itself (fraud, a timeout path). Robust summaries let you see the bulk clearly; the decision about the tail is a domain question.

<div class="take">
<div class="lab">Key takeaway</div>

report the median and IQR or MAD alongside the mean and standard deviation for any skewed metric, and screen outliers with order-statistic fences, never with z-scores computed from the same contaminated data.

</div>
