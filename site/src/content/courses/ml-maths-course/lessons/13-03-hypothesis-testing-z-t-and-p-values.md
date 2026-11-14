---
id: "13.03"
course: "ml-maths-course"
module: "13"
slug: "hypothesis-testing-z-t-and-p-values"
title: "Hypothesis testing: z, t and p-values"
keyIdeas: ["A hypothesis test asks how surprising the data would be if the null hypothesis $H_0$ were true; the **p-value** is the probability, under $H_0$, of a statistic at least as extreme as the one observed.", "Most test statistics are (estimate minus null value) divided by the standard error. The z-test uses a known or large-sample SE; the t-test uses an estimated SE and the heavier-tailed $t_{n-1}$ distribution, which matters below a few dozen observations.", "Pick the test to match the design: one-sample, paired (one-sample test on differences within units, which cancels shared variation) or two-sample Welch (independent groups, unequal variances, the safe default).", "Type I error (false positive) is capped by $\\alpha$; Type II error $\\beta$ shrinks as the effect, the sample size or the signal-to-noise ratio grows; power is $1-\\beta$. Underpowered tests miss effects and exaggerate the ones they find.", "A two-sided test rejects exactly when the null value is outside the confidence interval, so report the estimate with its interval: a p-value is not the probability that $H_0$ is true, not an effect size, and $p > \\alpha$ is not evidence of no effect."]
question: "Two models are scored on the same 40 test prompts. A Welch two-sample t-test on the two score columns gives p = 0.30, while a paired t-test gives p = 0.004. Explain how both can be correct calculations and which one answers the question. What else would you report or check before telling your team the new model is better?"
answer: "Both are arithmetically right but they test different designs. Welch treats the two columns as independent groups, so its standard error $\\sqrt{s_1^2/n_1 + s_2^2/n_2}$ includes the large prompt-to-prompt variation: some prompts are easy for both models and some are hard for both. That shared variation swamps a small, consistent difference, giving p = 0.30. The paired test works on per-prompt differences. Prompt difficulty affects both models alike, so it cancels in the subtraction ($\\mathrm{Var}(a-b) = \\mathrm{Var}(a) + \\mathrm{Var}(b) - 2\\,\\mathrm{Cov}(a,b)$ with a large positive covariance), the SE of the mean difference is far smaller, and the same mean gap becomes clearly significant. Since both models saw the same prompts, the observations are paired and the paired test (`scipy.stats.ttest_rel`, equivalently a one-sample t-test of the differences against 0 with 39 degrees of freedom) is the one that answers the question; Welch's independence assumption is false here and throws away the design. Before reporting: give the mean difference with its 95% confidence interval so the team sees how large the gain might be, not just that it is non-zero; look at the distribution of the 40 differences to check the result is not driven by a few prompts (a permutation or sign test is a useful robustness check at this size); confirm the test direction and the metric were fixed before looking; and remember that 40 prompts may not represent production traffic, so the interval describes this test set only."
sources: [{"title": "SciPy: scipy.stats.ttest_ind", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html", "note": "Welch versus Student, alternatives, and the permutation option."}, {"title": "Greenland et al.: Statistical tests, P values, confidence intervals, and power", "url": "https://link.springer.com/article/10.1007/s10654-016-0149-3", "note": "Twenty-five common misinterpretations of p-values and intervals, explained."}, {"title": "Penn State STAT 500: Applied Statistics", "url": "https://online.stat.psu.edu/stat500/", "note": "Lessons on z and t tests, paired designs and errors, with worked examples."}]
practice: ["z-test-one-sample", "t-test-one-sample", "one-sample-t-statistic"]
tags: ["p-value", "t-test", "power"]
tools: ["SciPy"]
---

## What you will learn

- The five-step logic shared by every classical hypothesis test, and what each step is for.
- How to compute a z statistic and a t statistic by hand, and why small samples need the t distribution.
- What a **p-value** measures, how to read it off a picture, and the four things it does *not* tell you.
- How to choose between one-sample, paired and two-sample (Welch) t-tests, with SciPy code for each.
- Type I and Type II errors, power, and why a test and a confidence interval are two views of one calculation.

*Before you start:* you need the standard error and the central limit theorem from lesson 13.02, and the mean and standard deviation from 13.01. Confidence intervals were introduced in the Inference module (lesson 12.03); a quick reminder is included below.

## From error bars to a yes/no question

Lesson 13.02 gave every statistic a standard error: the typical size of the noise in a number computed from a sample. A hypothesis test puts that noise to work on one specific question: *if nothing interesting were going on, how often would we see a result at least this far from "nothing"?* If the answer is "rarely", the data are hard to explain as noise alone.

The vocabulary built here (null, p-value, alpha, power) is reused for counts (13.04), resampling (13.05) and A/B experiments (13.06).

## The idea in one picture

You have built a new search ranker and want to know if it beats the old one. Both rankers are scored with NDCG (a 0-100 ranking quality score) on the same 25 test queries, and for each query you take the difference, new minus old.

| Quantity | Value
| --- | --- |
| Number of queries $n$ | 25
| Mean difference $\bar d$ | 1.2 points
| Standard deviation of the differences $s_d$ | 3.0 points
| Standard error $s_d/\sqrt n$ | 0.6 points

The new ranker is ahead by 1.2 points on average, but some queries got much better and some got worse. Is 1.2 a real gain, or the luck of these 25 queries?

~~~dg-fig 13.03-idea.svg
Left: the 25 per-query differences; most sit right of zero but several are negative. Right: the distribution the t statistic would have if the ranker gave no gain. The observed t = 2.0 sits in the tail, and the shaded area on both sides (the p-value) is 0.057.
~~~

The recipe: divide the effect by its standard error, $t = 1.2 / 0.6 = 2.0$. The mean gain is two standard errors away from zero. Then ask how often pure noise produces a value at least that far out, in either direction. The right panel answers it: if the ranker truly made no difference, about 5.7% of 25-query samples would give $|t| \ge 2$. That 0.057 is the **p-value**.

In plain words: a gain this big is somewhat unusual under "no gain", but not rare enough to clear the conventional 5% bar. The data are consistent with no gain and with a gain of more than 2 points. The honest verdict is "not yet clear", not "the ranker does nothing".

## Step 1: write down the null and the alternative

The **null hypothesis** $H_0$ is the boring explanation you are trying to rule out: no difference, or a mean equal to some reference value $\mu_0$. Here $H_0$: the mean per-query gain $\mu_d = 0$.

The **alternative** $H_1$ is what you conclude if the null is rejected. It is either:

- **two-sided**, $\mu \ne \mu_0$: you care about a change in either direction. This is the default.
- **one-sided**, $\mu > \mu_0$: only an improvement matters, and you committed to that before seeing data.

A one-sided test puts the whole 5% in one tail, so its p-value is half the two-sided one when the effect points the expected way (0.028 for the ranker instead of 0.057). Choosing one-sided *after* seeing which way the data went is a quiet way of doubling your false-positive rate, so decide in advance and write it down.

You also fix the **significance level** $\alpha$ in advance: the false-positive rate you will tolerate, conventionally 0.05.

## Step 2: compute a test statistic

A **test statistic** is a single number that is near zero when the data agree with $H_0$ and large when they disagree. Almost every test you will meet has the same shape:

$$\text{statistic} = \frac{\text{estimate} - \text{null value}}{\text{standard error of the estimate}}.$$

It measures the effect in units of its own noise: 1.2 points is interesting when the noise is 0.6, and not when it is 5.

**The z statistic.** If you know the population standard deviation $\sigma$ (say, from years of logs), or the sample is large enough that the sample estimate is very close to it,

$$z = \frac{\bar x - \mu_0}{\sigma / \sqrt n},$$

where $\bar x$ is the sample mean, $\mu_0$ the value claimed by the null, and $n$ the sample size. Under $H_0$, and thanks to the CLT from 13.02, $z$ follows the standard normal distribution.

**The t statistic.** Usually $\sigma$ is unknown and you plug in the sample standard deviation $s$:

$$t = \frac{\bar x - \mu_0}{s / \sqrt n}.$$

This looks like a tiny change, but it matters for small samples (Step 3).

<div class="ex">
<div class="lab">Worked example</div>

**a z-test on latency**

A service has a long-run mean latency of 200 ms with a well-established standard deviation $\sigma = 40$ ms. After a config change you sample $n = 64$ requests and get $\bar x = 211$ ms. Did latency change?

$H_0$: $\mu = 200$. Two-sided $H_1$: $\mu \ne 200$. $\alpha = 0.05$.

Standard error: $\sigma/\sqrt n = 40/\sqrt{64} = 40/8 = 5$ ms.

Statistic: $z = (211 - 200)/5 = 11/5 = 2.2$.

p-value: the standard normal puts $0.0139$ of its mass above 2.2, so both tails together hold $2 \times 0.0139 = 0.0278$.

Decision: $0.028 \le 0.05$, so reject $H_0$. The 95% interval $211 \pm 1.96 \times 5 = [201.2, 220.8]$ ms excludes 200 and says the slowdown is likely between 1 and 21 ms.

</div>

## Step 3: know the null distribution (why t is not z)

The **null distribution** is the distribution of the statistic if $H_0$ were true and you repeated the experiment many times. For $z$ it is the standard normal. For $t$ there is a catch: $s$ is itself estimated from the same small sample, so sometimes it comes out too small by chance, and then $t$ is blown up. That extra wobble gives $t$ heavier tails than the normal.

William Gosset (publishing as "Student") worked out the exact answer for normally distributed data:

$$t = \frac{\bar x - \mu_0}{s/\sqrt n} \sim t_{n-1},$$

the **Student t distribution** with $\nu = n - 1$ **degrees of freedom**. The degrees of freedom are $n$ values minus the one used up by estimating the mean.

~~~dg-fig 13.03-tz.svg
The t distribution is a normal with fatter tails; the fewer the observations, the fatter. Fatter tails push the 5% cutoff outward, from 1.96 for z to 2.26 at n = 10 and 3.18 at n = 4.
~~~

The two-sided 5% **critical value** (the cutoff that $|t|$ must beat) shrinks towards 1.96 as $n$ grows:

| Sample size $n$ | 4 | 10 | 25 | 50 | 100 | large (z)
| --- | --- | --- | --- | --- | --- | --- |
| Critical value | 3.18 | 2.26 | 2.06 | 2.01 | 1.98 | 1.96

The correction matters below about 20 observations and hardly beyond 50. Using z on the 25-query ranker would have given $p = 0.046$ instead of 0.057, flipping the verdict at $\alpha = 0.05$: exactly the overconfidence the t distribution exists to prevent.

The t-test assumes normal data, but the CLT makes it fairly robust at moderate $n$. It is *not* robust to strong skew or heavy outliers at small $n$; there, use a permutation test or bootstrap (13.05) or a rank-based test.

## Step 4: turn the statistic into a p-value

The **p-value** is the probability, computed under $H_0$, of getting a statistic at least as extreme as the one observed. "Extreme" means "in the direction(s) named by $H_1$": both tails for a two-sided test, one tail for a one-sided test.

In the ranker example the observed $t = 2.0$ with 24 degrees of freedom. The area of $t_{24}$ above 2.0 is 0.0285, so the two-sided p-value is $2 \times 0.0285 = 0.057$.

When $H_0$ is true, p-values are **uniformly distributed** between 0 and 1. Every value is equally likely, so a p-value below 0.05 happens exactly 5% of the time by construction. When there is a real effect, p-values bunch up near zero, but how tightly depends on the sample size.

~~~dg-fig 13.03-pvals.svg
Left: 20,000 simulated studies with n = 20. When the null is true (outline), p-values are flat and 4.9% fall below 0.05. With a real effect of 0.3 standard deviations, only 25% do. Right: among those studies, the ones that reach significance estimate the effect at 0.57 on average, nearly double the true 0.3.
~~~

## Step 5: decide, and say what you decided

Reject $H_0$ if $p \le \alpha$; otherwise "fail to reject". Failing to reject is not showing $H_0$ is true. For the ranker, $p = 0.057 > 0.05$, so you do not reject, and you report the estimate with its interval (below) so the reader can see how much is still uncertain.

~~~dg-flow Every classical test is the same pipeline; only the statistic and its null distribution change.
Data -(estimate / SE)-> Test statistic -(compare to)-> Null distribution -(tail area)-> *p-value -(vs alpha)-> Decision
~~~

<div class="ex">
<div class="lab">Worked example</div>

**is the new ranker better on 25 paired queries?**

Per-query differences in NDCG: mean $\bar d = 1.2$ points, standard deviation $s_d = 3.0$, $n = 25$.

Standard error: $\mathrm{SE} = 3.0/\sqrt{25} = 3.0/5 = 0.6$.

Statistic: $t = 1.2/0.6 = 2.0$ with $25 - 1 = 24$ degrees of freedom.

p-value: area of $t_{24}$ beyond $\pm 2.0$ is $2 \times 0.0285 = 0.057$. Not significant at 0.05.

95% interval: the critical value of $t_{24}$ is 2.064, so $1.2 \pm 2.064 \times 0.6 = 1.2 \pm 1.24$, i.e. $[-0.04, 2.44]$ points.

Reading: zero is just inside the interval, which is the same fact as $p$ being just above 0.05. The data are compatible with no gain and with a gain of over 2 points. The right next step is more queries, not a verdict that the ranker does nothing.

</div>

## Choosing the test for your design

The arithmetic above is a **one-sample t-test** applied to differences. Three designs cover most ML comparisons, and picking the wrong one is the commonest real-world error.

- **One-sample**: one column of numbers against a fixed value. Is the mean residual of my model zero? Is the mean latency 200 ms?
- **Paired**: two measurements on the *same* units: two models on the same prompts, the same users before and after. Take per-unit differences and run a one-sample test on them against 0.
- **Two-sample**: two *independent* groups, such as users randomly split into A and B. **Welch's t-test** uses $\mathrm{SE} = \sqrt{s_1^2/n_1 + s_2^2/n_2}$, where $s_1, s_2$ are the two groups' standard deviations and $n_1, n_2$ their sizes, plus an approximate degrees of freedom. It stays valid when the two groups have different spreads, so use it by default instead of the classic equal-variance (Student) version.

Why does pairing matter? Prompt difficulty hits both models alike; a two-sample test treats it as noise, while the paired test cancels it by subtracting within each prompt.

~~~dg-fig 13.03-paired.svg
Simulated scores for two models on 40 prompts. Left: prompt difficulty spreads scores from 40 to 80, so a Welch test that ignores pairing gives p = 0.69. Right: the same data as per-prompt differences have a spread of about 2 points, and the paired test gives p = 0.008.
~~~

In symbols, $\mathrm{Var}(\text{new} - \text{old}) = \sigma_{\text{new}}^2 + \sigma_{\text{old}}^2 - 2\,\mathrm{Cov}(\text{new}, \text{old})$. Shared difficulty makes the covariance large and positive, so the variance of the difference is far smaller than either variance alone. In the figure the scores have a standard deviation of about 10 points while the differences have about 2.

## Hypothesis tests in code

SciPy has all three tests. The code below simulates 40 prompts with a true gain of 1.5 points.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
difficulty = rng.normal(60, 10, 40)               # some prompts are just harder
old = difficulty + rng.normal(0, 1.5, 40)
new = difficulty + 1.5 + rng.normal(0, 1.5, 40)   # true gain: 1.5 points
d = new - old
print(d.mean(), d.std(ddof=1))                    # 1.053  2.116

res = stats.ttest_rel(new, old)                   # paired: the right test here
print(res.statistic, res.pvalue, res.df)          # 3.146  0.0032  39
print(stats.ttest_1samp(d, 0.0).pvalue)           # 0.0032  (identical: it IS a one-sample test on d)
print(res.confidence_interval(0.95))              # (0.376, 1.730)

print(stats.ttest_rel(new, old, alternative='greater').pvalue)   # 0.0016  one-sided, if pre-registered

w = stats.ttest_ind(new, old, equal_var=False)    # Welch: wrong design here, ignores pairing
print(w.pvalue, w.df)                             # 0.566  77.9
```

Note `ddof=1` for the sample standard deviation (divide by $n-1$), and that SciPy's default `equal_var=True` in `ttest_ind` is the Student test, so pass `equal_var=False` explicitly for Welch. For a z-test with known $\sigma$, compute `z = (xbar - mu0) / (sigma / np.sqrt(n))` and `p = 2 * stats.norm.sf(abs(z))`.

## Errors and power

Every test can be wrong in two ways.

| Decision | $H_0$ true | $H_0$ false
| --- | --- | --- |
| Reject $H_0$ | Type I error (false positive), rate $\alpha$ | Correct: power $1-\beta$
| Keep $H_0$ | Correct | Type II error (missed effect), rate $\beta$

A **Type I error** is declaring an effect that is not there; its rate is exactly $\alpha$, which you choose. A **Type II error** is missing a real effect; its rate $\beta$ is not chosen directly. **Power**, $1-\beta$, is the probability of detecting an effect of a given size. It goes up when the true effect is larger, the noise is smaller, or $n$ is larger.

~~~dg-fig 13.03-power.svg
A one-sided z-test with α = 0.05 rejects when z > 1.645. If the true effect is 2.5 standard errors, the statistic is centred at 2.5 instead of 0; the part of that curve left of the cutoff (β = 0.20) is missed, so power is 0.80.
~~~

Move the cutoff right (smaller $\alpha$) and you shrink false positives but grow $\beta$. The only way to shrink both is to push the two curves apart, which means a larger effect relative to its SE, i.e. more data or less noise. Lesson 13.06 turns this into a sample-size formula.

Underpowered studies have two problems. They usually miss real effects (only 25% power in the left panel of the p-value figure). And the effects they *do* flag are exaggerated, because only the lucky overestimates cross the threshold: the right panel of that figure shows significant estimates averaging 0.57 against a true 0.3. This is the **winner's curse**.

## Confidence intervals and tests are the same object

A 95% **confidence interval** for a mean is $\bar x \pm t_{n-1,\,0.975}\cdot s/\sqrt n$: the range of null values that the data do not reject. Precisely, a two-sided test at level $\alpha$ rejects $\mu = \mu_0$ exactly when $\mu_0$ lies outside the $1-\alpha$ interval. The ranker interval $[-0.04, 2.44]$ just includes 0, and $p$ is just above 0.05: one fact, said twice.

~~~dg-fig 13.03-ci.svg
Three results with their 95% intervals. A tiny effect can be "significant" and a large one "not significant"; the interval shows both whether zero is plausible and how big the effect might be.
~~~

A "significant" gain of 0.1 points with an interval of 0.01 to 0.19 and a "non-significant" gain of 3 points with an interval of -1 to 7 call for very different actions: the first is real but tiny; the second might be large and deserves a bigger test.

## What a p-value is not

- **Not the probability that $H_0$ is true.** It is computed *assuming* $H_0$. Turning it into $P(H_0 \mid \text{data})$ needs a prior (see Going deeper).
- **Not the size or importance of an effect.** With $n = 10^7$ users, a trivial difference gives $p < 10^{-6}$.
- **Not evidence of no effect when $p > \alpha$.** An underpowered test that fails to reject has simply not learned much.
- **Not valid after data-driven choices.** If you picked the test, the subgroup, the metric or the stopping time after looking, the 5% guarantee is gone. Lesson 13.06 covers peeking and multiple comparisons.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Two-sample test on paired data | Huge SE, p near 1 even with a consistent gain | Test per-unit differences (`ttest_rel`)
| Student instead of Welch | Wrong p when group sizes and spreads differ | `equal_var=False` by default
| z on a small sample | p-values too small, too many "wins" | Use t (critical 2.26 at n = 10, not 1.96)
| One-sided chosen after looking | False-positive rate silently doubles | Fix the direction in advance, or use two-sided
| Non-independent units | Thousands of "samples" that are really a few users | Test at the level of the randomised unit
| Small n, heavy skew or outliers | Result flips when one point is removed | Permutation or bootstrap test (13.05), or a rank test
| Bare p-value reported | Readers cannot tell tiny from large effects | Report estimate, interval and n

## Going deeper

**Where the t distribution comes from.** For normal data, $\bar x$ and $s^2$ are independent, $(\bar x - \mu)/(\sigma/\sqrt n)$ is standard normal, and $(n-1)s^2/\sigma^2$ follows a chi-squared distribution with $n-1$ degrees of freedom. The t distribution is by definition a standard normal divided by the square root of an independent chi-squared over its degrees of freedom; $\sigma$ cancels, which is why the test needs no knowledge of it. As $\nu \to \infty$ the chi-squared term concentrates at 1 and $t_\nu \to N(0,1)$.

**Welch's degrees of freedom.** With $v_i = s_i^2/n_i$, the Welch-Satterthwaite approximation is $\nu \approx (v_1+v_2)^2 / \big(v_1^2/(n_1-1) + v_2^2/(n_2-1)\big)$. It is usually not an integer (77.9 in the code above).

**Why p is uniform under the null.** If $T$ has a continuous null CDF $F$, the one-sided p-value is $1-F(T)$, and $F(T)$ is uniform by the probability integral transform. That is exactly what makes "$p \le \alpha$" a level-$\alpha$ rule.

**How often is a significant result false?** Suppose only 10% of the ideas you test have a real effect, tests run at $\alpha = 0.05$ with 80% power. Out of 1000 ideas, 900 null ideas give $0.05 \times 900 = 45$ false positives and 100 real ones give $0.8 \times 100 = 80$ true positives, so $45/125 = 36\%$ of your "significant" results are false. The p-value never told you this; the prior and the power did.

## Key terms

| Term | Meaning
| --- | --- |
| Null hypothesis $H_0$ | The "nothing going on" claim the test tries to rule out
| Test statistic | Effect divided by its standard error; large when data disagree with $H_0$
| Null distribution | The distribution of the statistic if $H_0$ were true
| p-value | Probability under $H_0$ of a statistic at least as extreme as the observed one
| Significance level $\alpha$ | Pre-chosen false-positive rate; reject when $p \le \alpha$
| Degrees of freedom | Information left for estimating spread; $n-1$ for a one-sample t-test
| Paired test | One-sample test on per-unit differences; cancels shared variation
| Welch's t-test | Two-sample test that allows unequal variances; the safe default
| Power $1-\beta$ | Probability of detecting a real effect of a given size

<div class="take">
<div class="lab">Key takeaway</div>

a test measures how surprising the data would be under the null, so match the statistic to the design (paired, Welch, one-sample), fix alpha and the hypotheses before looking, use t rather than z for small samples, and always report the effect with its confidence interval rather than a bare p-value.

</div>
