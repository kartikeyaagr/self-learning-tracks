---
id: "13.01"
course: "ml-maths-course"
module: "13"
slug: "describing-data-location-spread-and-shape"
title: "Describing data: location, spread and shape"
keyIdeas: ["Location answers \"where is the typical value\": the **mean** uses every value and can be dragged anywhere by one bad value, while the **median** only uses the order, so it cannot be moved past the middle by less than half the data.", "Spread comes in a fragile form (variance and standard deviation, dividing by $n-1$ for samples) and robust forms (IQR, and MAD scaled by 1.4826 to match $\\sigma$ for normal data). NumPy defaults to `ddof=0`, pandas to `ddof=1`.", "A quantile is a choice of rule: NumPy's default `linear` is Hyndman-Fan type 7. On small samples different rules can disagree wildly (22 to 480 ms for one p90), so state the rule and get more data.", "Skewness and excess kurtosis summarise shape; latency, revenue and counts are usually right-skewed and heavy-tailed, which is exactly when means and 3-sigma rules mislead.", "A z-score outlier rule can hide the outlier it is looking for (masking), because the outlier inflates the standard deviation. Use median and IQR or MAD fences, and inspect before deleting."]
question: "A teammate flags outliers in a column of per-user purchase amounts by removing rows with |z| > 3, recomputes the mean and standard deviation, and repeats until nothing is removed. What can go wrong, and what would you do instead?"
answer: "Purchase amounts are right-skewed with a heavy tail, so the first pass is distorted by masking: the biggest values inflate both the mean and the standard deviation, so their own z-scores shrink and several large values can hide each other; some genuine outliers (or data errors) survive. Iterating then has the opposite failure: each pass removes the largest values, which shrinks $s$, so ordinary large but legitimate purchases start crossing the threshold on the next pass. The procedure keeps eating the right tail, biasing the mean and total revenue downwards and making the data look far more normal than they are. (In small groups it can also do nothing at all: with $n \\le 10$ no value can reach $|z| > 3$, since the maximum is $(n-1)/\\sqrt{n}$.) Instead, compute fences from order statistics that a minority of outliers cannot distort: Tukey's $Q_3 + 1.5\\,\\mathrm{IQR}$, or a robust z-score $(x - \\text{median})/(1.4826\\,\\mathrm{MAD})$ with a cutoff near 3.5, possibly on the log of the amount since purchases are roughly log-normal. Then inspect what is flagged. Remove only values shown to be errors (test accounts, duplicated orders, currency bugs); for genuine big spenders, keep them, report a robust summary (median, trimmed mean, percentiles) alongside the mean, and winsorize or use a robust loss for modelling rather than deleting revenue that really happened."
sources: [{"title": "NIST/SEMATECH e-Handbook: Measures of Skewness and Kurtosis", "url": "https://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm", "note": "Definitions and the effect of heavy tails on these moments."}, {"title": "NIST/SEMATECH e-Handbook: Detection of Outliers", "url": "https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm", "note": "Why masking happens and the modified (robust) z-score rule."}, {"title": "Hyndman and Fan: Sample quantiles in statistical packages", "url": "https://robjhyndman.com/publications/quantiles/", "note": "The nine quantile definitions that NumPy and R implement."}, {"title": "NumPy: numpy.quantile", "url": "https://numpy.org/doc/stable/reference/generated/numpy.quantile.html", "note": "The method argument and which Hyndman-Fan type each one is."}]
practice: ["percentiles-quantiles", "skewness-and-kurtosis", "mean-median-mode", "sample-variance-and-standard-deviation", "winsorize"]
tags: ["robust statistics", "quantiles"]
tools: ["NumPy", "SciPy"]
---

## What you will learn

- The three questions to ask of any column of numbers: where is it (**location**), how wide is it (**spread**) and what does it look like (**shape**).
- How the mean, median and trimmed mean differ, and why one bad value can move the mean anywhere but cannot drag the median.
- Why the sample variance divides by $n-1$, and the robust spread measures (IQR and MAD) that can stand in for a standard deviation.
- Why a quantile such as "p90 latency" depends on a rule you choose, and how much that rule matters on small samples.
- How to spot skew and heavy tails, and how to screen outliers without the outliers hiding themselves.

*Before you start:* you need averages, sums and absolute values. Lesson 12.01 (estimators, bias and variance) explains what "unbiased" means; it helps for Step 3 but is not essential. The code uses NumPy, SciPy and pandas.

## Where this module starts

Modules 10 to 12 treated random variables and estimators in theory. This module is about the numbers you actually hold: a column of request latencies, a list of per-user revenues, a vector of model errors. This lesson summarises a sample honestly; 13.02 measures how much those summaries wobble, 13.03 to 13.05 turn the wobble into tests, and 13.06 combines everything in an A/B experiment. Every later lesson rests on a few summary statistics, so it pays to know how each reacts to messy data. The recurring theme: each question has a **fragile** answer that gives every value equal weight, and a **robust** answer that relies only on the order of the values.

## The idea in one picture

A service logs ten requests. Their latencies, in milliseconds, sorted:

| Request | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Latency (ms) | 12 | 13 | 13 | 14 | 15 | 15 | 16 | 18 | 22 | 480

Nine requests took between 12 and 22 ms. One hit a slow path and took 480 ms. What is the "typical" latency?

~~~dg-fig 13.01-latency.svg
The median (15 ms) sits inside the cluster of nine fast requests. The mean (61.8 ms) sits in empty space: no request took anywhere near that long.
~~~

- The **mean** (the ordinary average) is $618/10 = 61.8$ ms.
- The **median** (the middle value once sorted) is 15 ms.

The mean describes no actual request: it is the balance point, and one value 400 ms away drags it out of the cluster. The median only asks which value is in the middle, so the slow request is just "one value on the high side", however large.

Neither number is wrong; they answer different questions. The mean gives total cost per request (ten requests cost 618 ms of server time). The median gives what a typical user experiences. Know which question you are asking, and report both when the data are lopsided.

## Step 1: location, where the typical value is

Write the sample as $x_1, \dots, x_n$, and the same values sorted as $x_{(1)} \le x_{(2)} \le \dots \le x_{(n)}$ (the **order statistics**).

**Mean.** Add up and divide by the count:

$$\bar x = \frac{1}{n}\sum_{i=1}^{n} x_i.$$

It weighs every value equally, is best for clean symmetric data, and is the only location statistic that "adds up" (total revenue = mean revenue × users).

**Median.** The middle order statistic, $x_{((n+1)/2)}$ when $n$ is odd, and the average of the two middle ones when $n$ is even. For our ten latencies the 5th and 6th values are both 15, so the median is 15.

The mean is the $m$ that minimises total *squared* distance $\sum_i (x_i - m)^2$; the median minimises total *absolute* distance $\sum_i |x_i - m|$. Squaring makes a far-away point expensive, so the mean moves towards it.

**Trimmed mean.** Sort, drop the top and bottom $k\%$, average the rest. The 10% trimmed mean of our data drops 12 and 480 and averages the remaining eight: $126/8 = 15.75$ ms.

How robust is each one? The **breakdown point** is the fraction of the data you can replace with garbage before the statistic can be pushed arbitrarily far. Drag the slowest request from 22 ms up to 1000 ms and watch:

~~~dg-fig 13.01-breakdown.svg
Dragging one value from 22 to 1000 ms. The mean and standard deviation follow it without limit (log scale); the median, trimmed mean, IQR and MAD do not move at all.
~~~

At 1000 ms the mean is $(138 + 1000)/10 = 113.8$ ms while the median is still 15. One value out of ten broke the mean, so its breakdown point is $0$ (in the limit of large $n$, a single value always suffices). A 10% trimmed mean survives up to 10% garbage; the median survives up to 50%, the most any location statistic can.

## Step 2: spread, how wide the data are

Two samples can share a median and still differ hugely in width. Spread measures that.

**Variance and standard deviation.** Take each value's deviation from the mean, square it, average:

$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar x)^2, \qquad s = \sqrt{s^2}.$$

Here $x_i - \bar x$ is the deviation of value $i$, $s^2$ is the **sample variance** (in squared units, ms²) and $s$ is the **standard deviation** (back in ms). Ignore the odd $n-1$ for a moment; Step 3 explains it.

Number check on our latencies: the deviations from 61.8 are $-49.8, -48.8, \dots, -39.8$ for the nine fast requests and $+418.2$ for the slow one. Their squares sum to 194 399.6. Divide by 9: $s^2 \approx 21\,600$, so $s \approx 147.0$ ms. The slow request contributes $418.2^2 \approx 174\,900$ of that sum, about 90%.

The robust alternatives use order instead of squares:

- **IQR** (interquartile range) $= Q_3 - Q_1$, the width of the middle half of the data. $Q_1$ and $Q_3$ are the 25th and 75th percentiles (Step 4 shows how to compute them).
- **MAD** (median absolute deviation) $= \mathrm{median}_i\,|x_i - \mathrm{median}(x)|$: the typical distance from the median, measured robustly.

Neither equals a standard deviation directly, but for normal data both can be rescaled to one: $\sigma \approx 1.4826 \cdot \mathrm{MAD}$ and $\sigma \approx \mathrm{IQR}/1.349$. (The constants come from the normal distribution: its quartiles sit $0.6745\sigma$ either side of the centre, and $1/0.6745 = 1.4826$, $2 \times 0.6745 = 1.349$.)

| Statistic | Measures | Breakdown point | Use when
| --- | --- | --- | --- |
| Mean | location | 0% | data are clean and roughly symmetric, or totals matter
| Trimmed mean (10%) | location | 10% | a compromise with a few wild values
| Median | location | 50% | skewed data, outliers, latency
| Standard deviation | spread | 0% | feeding normal-theory formulas
| IQR | spread | 25% | box plots, outlier fences
| MAD | spread | 50% | robust z-scores

## Step 3: why divide by n minus 1

The true variance $\sigma^2$ is the average squared distance from the *true* mean $\mu$. We do not know $\mu$, so we measure distances from $\bar x$ instead. But $\bar x$ was computed from these very points, and it is the number that makes $\sum (x_i - m)^2$ as small as possible. So squared distances from $\bar x$ are systematically a little *smaller* than squared distances from $\mu$. Dividing by $n$ would therefore underestimate $\sigma^2$, on average by the factor $(n-1)/n$. Dividing by $n-1$ instead (**Bessel's correction**) exactly cancels that, making $s^2$ an **unbiased** estimator: right on average.

Equivalently, once $\bar x$ is known only $n-1$ deviations are free (they must sum to zero): the **degrees of freedom**.

~~~dg-fig 13.01-bessel.svg
Average variance estimate over 40 000 simulated normal samples with true variance 1. Dividing by n lands at (n−1)/n, only 0.5 for n = 2; dividing by n−1 is on target at every sample size.
~~~

In the simulation, at $n = 2$ the divide-by-$n$ estimate averaged 0.503 against a true value of 1, while the $n-1$ version averaged 1.007. By $n = 20$ the gap is 5%; for thousands of rows it is negligible. It still matters for consistency between tools. NumPy's `np.std` and `np.var` default to `ddof=0` (divide by $n$); pandas' `.std()` and `.var()` default to `ddof=1`. Same data, different answers: 139.4 versus 147.0 ms on our latencies.

<div class="ex">
<div class="lab">Worked example</div>

**variance by hand, both ways**

Data: three values $2, 4, 9$. Mean: $\bar x = 15/3 = 5$.

Deviations: $2-5 = -3$, $4-5 = -1$, $9-5 = 4$. They sum to $0$, as deviations from the mean always do.

Squares: $9, 1, 16$, which sum to $26$.

Divide by $n = 3$: $26/3 \approx 8.67$. Divide by $n-1 = 2$: $26/2 = 13$. The sample variance is $s^2 = 13$ and $s = \sqrt{13} \approx 3.61$.

Check with the formula from the figure: on average the divide-by-$n$ version is $(n-1)/n = 2/3$ of the other, and indeed $8.67 = \tfrac{2}{3} \times 13$.

</div>

## Step 4: quantiles are a choice, not a fact

The **$p$-quantile** is a value with a fraction $p$ of the data below it: the median is the 0.5-quantile, $Q_1$ the 0.25-quantile, "p99 latency" the 0.99-quantile. With $n$ points, most values of $p$ land *between* two order statistics, so someone has to decide what to report.

NumPy's `np.quantile(x, p)` defaults to `method="linear"`, which is type 7 of the nine rules catalogued by Hyndman and Fan (R uses the same default):

1. Sort the data.
2. Compute the position $h = (n-1)\,p$, counting from 0.
3. Take the values at positions $\lfloor h \rfloor$ and $\lceil h \rceil$ and interpolate linearly between them.

<div class="ex">
<div class="lab">Worked example</div>

**quartiles and the 90th percentile of the ten latencies**

$Q_1$: $h = 9 \times 0.25 = 2.25$. Positions 2 and 3 (0-based) hold 13 and 14, so $Q_1 = 13 + 0.25 \times (14 - 13) = 13.25$.

$Q_3$: $h = 9 \times 0.75 = 6.75$. Positions 6 and 7 hold 16 and 18, so $Q_3 = 16 + 0.75 \times 2 = 17.5$. The IQR is $17.5 - 13.25 = 4.25$ ms.

p90: $h = 9 \times 0.9 = 8.1$. Positions 8 and 9 hold 22 and 480, so p90 $= 22 + 0.1 \times 458 = 67.8$ ms.

MAD: distances from the median 15 are $3, 2, 2, 1, 0, 0, 1, 3, 7, 465$. Sorted: $0, 0, 1, 1, 2, 2, 3, 3, 7, 465$; the middle two are both 2, so MAD $= 2$ and the robust scale is $1.4826 \times 2 \approx 2.97$ ms.

</div>

That p90 of 67.8 ms is a number no request ever had, produced by interpolating across a 458 ms gap. Other rules handle the gap differently:

~~~dg-fig 13.01-quantiles.svg
Left: the empirical CDF jumps from 0.9 to 1.0 between 22 and 480 ms, so the data do not pin down the 90th percentile. Right: seven NumPy quantile methods report anything from 22 to 480 ms for the same ten values.
~~~

On the same ten values, `lower` and `inverted_cdf` report 22 ms, `linear` 67.8, `hazen` 251, `median_unbiased` 312, `weibull` 434 and `higher` 480. No rule is at fault: ten values simply do not contain the information. State the rule, and prefer more data over a cleverer rule; a p99 from 100 values is just the largest one or two.

## Step 5: shape, skew and heavy tails

Shape asks: is the data lopsided, and how often do extremes occur? With the central moments $m_k = \frac{1}{n}\sum_i (x_i - \bar x)^k$ (so $m_2$ is the divide-by-$n$ variance):

$$g_1 = \frac{m_3}{m_2^{3/2}}, \qquad g_2 = \frac{m_4}{m_2^{2}} - 3.$$

- $g_1$ is the **skewness**. Cubing keeps the sign of each deviation, so a long right tail gives $g_1 > 0$ and usually puts the mean above the median.
- $g_2$ is the **excess kurtosis**. Raising to the fourth power makes it dominated by the most extreme values; the $-3$ makes it 0 for a normal distribution. Large positive values mean **heavy tails**: extremes far more often than a normal would produce.

~~~dg-fig 13.01-shape.svg
Three samples of 20 000 values with medians near 50. The log-normal is right-skewed (skew 2.5) and its mean is pulled to 59.6. The Student t looks almost normal in the middle, yet its excess kurtosis is 36: rare, huge values.
~~~

Latency, revenue, file sizes and counts are typically right-skewed and heavy-tailed, which is exactly when the mean and "3-sigma" rules mislead. For our ten latencies SciPy gives $g_1 = 2.66$ and $g_2 = 5.11$; without the 480 the nine remaining values have $g_1 = 1.14$ and $g_2 = 0.45$.

Third and fourth powers make both even more fragile than the variance: treat them as flags, not measurements. In SciPy, `stats.skew` and `stats.kurtosis` default to these divide-by-$n$, excess (Fisher) forms; pass `bias=False` for the adjusted versions and `fisher=False` for kurtosis without the $-3$.

## Step 6: outliers and the masking trap

Two common rules for flagging outliers:

1. **z-score**: flag $|x_i - \bar x|/s > 3$.
2. **Tukey fences**: flag values below $Q_1 - 1.5\,\mathrm{IQR}$ or above $Q_3 + 1.5\,\mathrm{IQR}$. The **robust z-score** $(x_i - \mathrm{median})/(1.4826 \cdot \mathrm{MAD})$ with a cutoff near 3.5 is the same idea on a different scale.

<div class="ex">
<div class="lab">Worked example</div>

**does each rule catch the 480 ms request?**

z-score: $(480 - 61.8)/147.0 = 2.85$. That is below 3, so the 3-sigma rule does **not** flag it.

Robust z-score: $(480 - 15)/2.97 \approx 157$. Flagged by a huge margin. The 22 ms request scores $(22-15)/2.97 \approx 2.4$, below 3.5, so it is kept.

Tukey: upper fence $17.5 + 1.5 \times 4.25 = 23.875$. 480 is flagged; 22 is inside. The lower fence $13.25 - 6.375 = 6.875$ flags nothing.

</div>

The 480 ms value inflated the mean (61.8 instead of 15.3) and the standard deviation (147 instead of 3.1): the yardstick grew with the thing measured. Several outliers hide each other even better. This is **masking**.

~~~dg-fig 13.01-masking.svg
Copies of a 480 ms outlier added to 100 normal requests (mean 15, sd 2). One outlier has z ≈ 9.9; with 12 or more, every one of them has z below 3 and none is flagged. The robust z-score stays above 100 throughout.
~~~

The fix is to build the yardstick from order statistics, which a minority of outliers cannot distort:

~~~dg-flow Robust outlier screening: every step uses order statistics, so the outlier cannot distort its own yardstick.
Raw sample -(sort)-> Q1, median, Q3 -(IQR or MAD)-> Robust scale -(fences)-> *Flag outliers -(inspect)-> Fix, keep or model
~~~

Flagging is not deleting. An outlier may be a logging bug (drop it), a rare real event (keep it; report a tail percentile), or the signal itself (fraud, a timeout path). That decision is a domain question.

## Describing data in code

```python
import numpy as np, pandas as pd
from scipy import stats

x = np.array([12, 13, 13, 14, 15, 15, 16, 18, 22, 480.0])

print(x.mean(), np.median(x), stats.trim_mean(x, 0.1))   # 61.8 15.0 15.75
print(x.std(), x.std(ddof=1), pd.Series(x).std())         # 139.43 146.97 146.97

q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75])            # 13.25 15.0 17.5
iqr = q3 - q1                                              # 4.25
mad = stats.median_abs_deviation(x)                        # 2.0
sigma_mad = stats.median_abs_deviation(x, scale="normal")  # 2.965  (= 1.4826 * MAD)

print(stats.skew(x), stats.kurtosis(x))                    # 2.66 5.11

z = (x - x.mean()) / x.std(ddof=1)
robust_z = (x - med) / sigma_mad
tukey = (x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)
print(z[-2:], robust_z[-2:], tukey.nonzero()[0])           # [-0.27 2.85] [2.4 156.8] [9]

print(pd.Series(x).describe())   # count, mean, std (ddof=1), min, 25%, 50%, 75%, max
```

`describe()` puts the mean next to the quartiles: here the mean (61.8) lies above the 75th percentile (17.5), a loud sign that a few large values dominate.

## Reading a summary

A quick checklist whenever someone hands you "mean = X, sd = Y":

- **Compare mean and median.** If the mean sits well above the median, the data are right-skewed; if it is above $Q_3$, a handful of values dominate.
- **Compare $s$ with the robust scale.** If $s$ is many times $1.4826 \cdot \mathrm{MAD}$ or $\mathrm{IQR}/1.349$ (here 147 versus about 3), the spread is driven by the tails, not the bulk.
- **Look at a histogram or box plot**, with a log axis for positive skewed data. Summary numbers can hide bimodality (two clusters) completely.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Reporting only the mean of a skewed metric | "average latency 62 ms" when 90% of requests take under 20 | report median and p90/p99 alongside the mean
| Mixing `ddof` defaults | NumPy and pandas disagree on the same column | set `ddof=1` explicitly for samples
| Quantiles from tiny samples | p99 equals the maximum; changes wildly between days | state the rule; collect more data; report a confidence interval (13.05)
| z-score outlier rule | big outliers not flagged (masking), or normal values flagged after repeated passes | Tukey fences or MAD-based robust z
| Deleting everything flagged | revenue or tail risk silently disappears | inspect first; winsorize or model the tail instead

## Going deeper

**Bounded z-scores.** With the $n-1$ divisor, no point in a sample of size $n$ can have $|z|$ larger than $(n-1)/\sqrt{n}$. For $n = 10$ that ceiling is 2.846, and our 480 ms request scored 2.845, essentially as extreme as any value can look. A 3-sigma rule therefore cannot flag *anything* in a sample of 10 or fewer; you need $n \ge 11$ (ceiling 3.015) before it is even possible. Similarly, the divide-by-$n$ skewness of $n$ points cannot exceed $(n-2)/\sqrt{n-1}$, which is $2.67$ for $n = 10$; our $g_1 = 2.66$ is at that limit.

**Efficiency versus robustness.** For truly normal data the median is less precise than the mean: its sampling variance is about $\pi/2 \approx 1.57$ times larger, so it "wastes" roughly a third of the data. For heavy-tailed data the order flips and the median wins by a lot. Trimmed means and M-estimators such as Huber's sit in between, near-mean efficiency on clean data with bounded influence from outliers.

**The influence function** measures how much one added point at value $x$ moves a statistic. For the mean it is $x - \mu$, unbounded; for the median it is a bounded step, since only the side of the median matters.

**Winsorizing.** Instead of dropping the extreme $k\%$, clip them to the $k$-th and $(100-k)$-th percentiles; the sample size is kept and the mean becomes robust. Common before modelling revenue-like targets.

## Key terms

| Term | Meaning
| --- | --- |
| Order statistics | The sample values sorted from smallest to largest
| Median | The middle order statistic; minimises total absolute distance
| Trimmed mean | The mean after dropping a fixed fraction from each end
| Breakdown point | Fraction of bad values a statistic tolerates before it can be pushed arbitrarily far
| Bessel's correction | Dividing by $n-1$ so the sample variance is unbiased
| IQR | $Q_3 - Q_1$, the width of the middle half; $\approx 1.349\sigma$ for normal data
| MAD | Median absolute distance from the median; $\times 1.4826$ estimates $\sigma$ for normal data
| Skewness / excess kurtosis | Third- and fourth-moment shape measures: lopsidedness and tail heaviness
| Masking | Outliers inflating the mean and sd so that they no longer look extreme

<div class="take">
<div class="lab">Key takeaway</div>

report the median and IQR or MAD alongside the mean and standard deviation for any skewed metric, state which quantile rule and which `ddof` you used, and screen outliers with order-statistic fences, never with z-scores computed from the same contaminated data.

</div>
