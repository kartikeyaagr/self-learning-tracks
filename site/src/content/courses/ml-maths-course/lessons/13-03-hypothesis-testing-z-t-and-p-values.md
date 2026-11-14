---
id: "13.03"
course: "ml-maths-course"
module: "13"
slug: "hypothesis-testing-z-t-and-p-values"
title: "Hypothesis testing: z, t and p-values"
keyIdeas: ["A test asks how surprising the data would be if the null hypothesis $H_0$ were true; the **p-value** is the probability, under $H_0$, of a statistic at least as extreme as the one observed.", "The z-test divides an effect by a known or large-sample SE; the t-test uses an estimated SE and the heavier-tailed $t_{\\nu}$ distribution, which matters below a few dozen observations.", "Pick the test to match the design: one-sample, paired (difference within units) or two-sample Welch (unequal variances, the safe default).", "Type I error (false positive) is capped by $\\alpha$; Type II error $\\beta$ depends on effect size, noise and $n$; power is $1-\\beta$.", "A p-value is not the probability that $H_0$ is true, and a non-significant result is not evidence of no effect; report the estimate with a confidence interval."]
question: "Two models are scored on the same 40 test prompts. A Welch two-sample t-test on the two score columns gives p = 0.30, while a paired t-test gives p = 0.004. Explain how both can be correct calculations and which one answers the question."
answer: "Both are arithmetically right but they test different designs. Welch treats the columns as independent groups, so its SE includes the large prompt-to-prompt variation: some prompts are easy for both models, some hard for both. That shared variation swamps a small, consistent difference, giving p = 0.30. The paired test works on per-prompt differences; prompt difficulty cancels because it affects both models, so the SE of the mean difference is far smaller and the same mean gap becomes clearly significant. Since both models saw the same prompts, the observations are paired and the paired test is the correct one; Welch's independence assumption is false here and wastes the design. You should still check that the differences are not dominated by a few prompts (look at their distribution, or use a permutation or sign test) and report the mean difference with its interval."
sources: [{"title": "SciPy: scipy.stats.ttest_ind", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html", "note": "Welch versus Student, alternatives, and the permutation option."}, {"title": "Greenland et al.: Statistical tests, P values, confidence intervals, and power", "url": "https://link.springer.com/article/10.1007/s10654-016-0149-3", "note": "Twenty-five common misinterpretations of p-values and intervals, explained."}, {"title": "Penn State STAT 500: Applied Statistics", "url": "https://online.stat.psu.edu/stat500/", "note": "Lessons on z and t tests, paired designs and errors, with worked examples."}]
practice: ["z-test-one-sample", "t-test-one-sample", "one-sample-t-statistic"]
tags: ["p-value", "t-test", "power"]
tools: ["SciPy"]
---

## The logic of a test

Lesson 13.02 gave every statistic a standard error. A hypothesis test uses it to ask one question: if nothing interesting were going on, how often would we see a result this far from "nothing"?

1. State a **null hypothesis** $H_0$ (no difference, mean equals $\mu_0$) and an **alternative** $H_1$ (two-sided $\mu\ne\mu_0$, or one-sided $\mu>\mu_0$, decided before looking).
2. Compute a **test statistic** that is large when the data disagree with $H_0$, usually (estimate minus null value) / SE.
3. Find its **null distribution**: the distribution it would have if $H_0$ were true.
4. The **p-value** is the probability, under that distribution, of a statistic at least as extreme as the observed one.
5. Reject $H_0$ if $p \le \alpha$, with $\alpha$ fixed in advance (commonly 0.05).

~~~dg-flow Every classical test is the same pipeline; only the statistic and its null distribution change.
Data -(estimate / SE)-> Test statistic -(compare to)-> Null distribution -(tail area)-> *p-value -(vs alpha)-> Decision
~~~

## z-tests and t-tests

With a known $\sigma$, or a large sample where $s\approx\sigma$, the statistic $z = (\bar x - \mu_0)/(\sigma/\sqrt n)$ is standard normal under $H_0$. With $\sigma$ estimated from a small sample, the extra uncertainty in $s$ makes the statistic heavier-tailed. Gosset showed that for normal data

$$t = \frac{\bar x - \mu_0}{s/\sqrt n} \sim t_{n-1}.$$

The two-sided 5% critical value is 2.26 at $n=10$, 2.06 at $n=25$ and 1.98 at $n=100$, against 1.96 for $z$; beyond roughly 50 observations the distinction rarely changes a decision. The t-test is also fairly robust to non-normality at moderate $n$ thanks to the CLT, but not to strong skew at small $n$.

Three designs cover most ML comparisons:

- **One-sample**: is the mean error zero? Test the column against $\mu_0$.
- **Paired**: two models evaluated on the same 25 queries, or the same users before and after. Take per-unit differences and run a one-sample t-test on them. Pairing removes between-unit variance and is often several times more powerful than ignoring it.
- **Two-sample**: independent groups. **Welch's** test uses $\mathrm{SE} = \sqrt{s_1^2/n_1 + s_2^2/n_2}$ and an approximate degrees of freedom; it stays valid when variances differ, so use it by default (`scipy.stats.ttest_ind(a, b, equal_var=False)`).

```
from scipy import stats
d = new_scores - old_scores                      # same 25 queries, paired
t, p = stats.ttest_1samp(d, 0.0)                 # == stats.ttest_rel(new, old)
t2, p2 = stats.ttest_ind(new_scores, old_scores, equal_var=False)  # wrong here: ignores pairing
```

## Errors and power

| Decision | $H_0$ true | $H_0$ false
| --- | --- | --- |
| Reject $H_0$ | Type I error (rate $\alpha$) | Correct: power $1-\beta$
| Keep $H_0$ | Correct | Type II error (rate $\beta$)

$\alpha$ is under your control. $\beta$ is not chosen directly; it falls as the true effect grows, as noise shrinks and as $n$ grows. Lesson 13.06 turns that relationship into a sample-size calculation. An underpowered study has two problems: it usually misses real effects, and the effects it does declare significant are exaggerated, because only the lucky overestimates cross the threshold.

## Confidence intervals and tests are the same object

A two-sided test at level $\alpha$ rejects $\mu = \mu_0$ exactly when $\mu_0$ lies outside the $1-\alpha$ confidence interval $\bar x \pm t_{n-1,\,1-\alpha/2}\cdot s/\sqrt n$ (interval estimation is covered in lesson 12.03). The interval is more informative: it shows both whether zero is plausible and how large the effect might be. A "significant" improvement of 0.1% with an interval of 0.01% to 0.19% and a "non-significant" one of 3% with an interval of -1% to 7% call for very different actions.

## What a p-value is not

- Not $P(H_0 \mid \text{data})$. That needs a prior; with a low prior on real effects, many results with $p\approx0.04$ are false positives.
- Not the size or importance of an effect. With $n = 10^7$, trivial differences are significant.
- Not evidence of no effect when $p > \alpha$. Absence of evidence in an underpowered test is just absence of evidence.
- Not valid after you chose the test, the subgroup or the stopping time by looking at the data. Lesson 13.06 deals with peeking and multiple comparisons.

<div class="ex">
<div class="lab">Worked example</div>

**is the new ranker better on 25 paired queries?**

Per-query differences in NDCG: mean $\bar d = 1.2$ points, standard deviation $s_d = 3.0$.

$\mathrm{SE} = 3.0/\sqrt{25} = 0.6$, so $t = 1.2/0.6 = 2.0$ with 24 degrees of freedom.

Two-sided $p = 0.057$: not significant at 0.05. The 95% interval is $1.2 \pm 2.064\cdot0.6$, roughly -0.04 to 2.44 points.

Reading: the data are compatible with no gain and with a gain of over 2 points. The right next step is more queries, not a verdict that the ranker does nothing.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

a test measures how surprising the data are under the null, so match the statistic to the design (paired, Welch, one-sample), fix alpha and the hypotheses before looking, and always report the effect with its confidence interval rather than a bare p-value.

</div>
