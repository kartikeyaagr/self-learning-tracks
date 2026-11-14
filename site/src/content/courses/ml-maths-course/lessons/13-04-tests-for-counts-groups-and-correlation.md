---
id: "13.04"
course: "ml-maths-course"
module: "13"
slug: "tests-for-counts-groups-and-correlation"
title: "Tests for counts, groups and correlation"
keyIdeas: ["Chi-square tests compare observed counts with expected counts: $\\chi^2 = \\sum (O-E)^2/E$, with $k-1$ degrees of freedom for goodness of fit and $(r-1)(c-1)$ for an $r\\times c$ independence table.", "For a $2\\times2$ table the uncorrected chi-square statistic equals the square of the two-proportion z statistic; when expected counts fall below about 5, use Fisher's exact test (and remember SciPy applies Yates' correction to $2\\times2$ tables by default).", "One-way ANOVA tests whether $k$ group means are all equal with $F = \\mathrm{MS_{between}}/\\mathrm{MS_{within}}$: gaps between means are judged against the noise within groups. A significant F says some mean differs, not which; Tukey's HSD says which.", "A Pearson correlation is tested with $t = r\\sqrt{n-2}/\\sqrt{1-r^2}$ on $n-2$ degrees of freedom; Fisher's $z = \\operatorname{atanh} r$ (SE $1/\\sqrt{n-3}$) gives intervals. Spearman's rank correlation handles monotone, non-linear or outlier-prone data.", "All of these assume independent observations; counting events instead of units (clicks instead of users) silently breaks them and can push the false-positive rate far above 5%."]
question: "You compare error-type distributions (5 error categories) between two model versions using every logged error: 12,000 errors from version 1 and 11,500 from version 2, collected from 300 users. The chi-square test of independence gives p = 1e-9. What would you check before concluding the versions fail differently?"
answer: "First, independence: the 23,500 errors come from only 300 users, and a few heavy users with their own peculiar inputs can generate thousands of correlated errors, so the chi-square treats clustered events as independent and the p-value is far too small (in an A/A simulation with 50 events per user, an event-level test rejects about 39% of the time instead of 5%). Re-run the analysis with the user as the unit, for example per-user error-type proportions compared with a permutation test that shuffles version labels across users, or a user-level bootstrap. Second, confounding: did the same users and traffic mix see both versions, or did version 2 roll out to different segments or later dates? A sample-ratio and segment-mix check (itself a goodness-of-fit test) helps here. Third, effect size: with 23,500 counts tiny shifts are significant, so inspect adjusted residuals per cell to see which categories moved and by how much, compute Cramér's V, and judge whether that matters. Finally, check expected counts in rare categories, merging them or using an exact test if any fall below about 5."
sources: [{"title": "SciPy: scipy.stats.chi2_contingency", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html", "note": "Expected frequencies, degrees of freedom and the Yates correction default."}, {"title": "NIST/SEMATECH e-Handbook: Chi-Square Goodness-of-Fit Test", "url": "https://www.itl.nist.gov/div898/handbook/eda/section3/eda35f.htm", "note": "Binning, expected counts and degrees of freedom."}, {"title": "SciPy: scipy.stats.f_oneway", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html", "note": "One-way ANOVA, its assumptions and the alternatives SciPy points to."}, {"title": "SciPy: scipy.stats.pearsonr", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html", "note": "The t-based test and Fisher-z confidence interval for r."}]
practice: ["chi-square-goodness-of-fit", "independence-testing", "anova-f-statistic", "correlation-significance-test"]
tags: ["chi-square", "ANOVA", "correlation"]
tools: ["SciPy"]
---

## What you will learn

- How to test counts with the chi-square statistic: goodness of fit (does one categorical variable match a target split?) and independence (are two categorical variables related?).
- Why a $2\times2$ chi-square test is the two-proportion z-test in disguise, and when to switch to Fisher's exact test.
- How one-way ANOVA compares several group means at once by splitting variation into "between groups" and "within groups".
- How to test a correlation, build a confidence interval for it with Fisher's z transform, and when Spearman's rank correlation is the better choice.
- Why every one of these tests silently breaks when you count events instead of independent units.

*Before you start:* you need the test pipeline from 13.03 (statistic, null distribution, p-value), the standard error and the CLT from 13.02, and the median, outliers and breakdown point from 13.01.

## Beyond one mean

Lesson 13.03 compared one or two means. Much ML data is instead **categorical** (did the user convert, which error class did the model produce), has more than two groups (four prompt templates), or asks whether two numeric columns move together. The logic does not change: a statistic that is small under the null, its null distribution, and a tail area as the p-value. Only the statistic changes.

| Question | Data | Test | Null distribution
| --- | --- | --- | --- |
| Do counts match a target distribution? | one categorical variable | chi-square goodness of fit | $\chi^2_{k-1}$
| Are two categorical variables related? | contingency table | chi-square independence, Fisher exact | $\chi^2_{(r-1)(c-1)}$
| Do several group means differ? | numeric outcome, k groups | one-way ANOVA (Welch ANOVA if variances differ) | $F_{k-1,\,N-k}$
| Are two numeric variables linearly related? | paired numeric columns | Pearson t-test, Fisher z | $t_{n-2}$
| Monotone relation, outliers, ranks? | paired columns | Spearman rank correlation | approx. $t_{n-2}$ or permutation

## The idea in one picture

You test a new onboarding flow. Each of 2,000 users is randomly shown variant A or variant B, 1,000 each, and you record whether they converted:

| Arm | Converted | Did not convert | Total
| --- | --- | --- | --- |
| Variant A | 200 | 800 | 1,000
| Variant B | 240 | 760 | 1,000
| Total | 440 | 1,560 | 2,000

This is a **contingency table**: counts cross-classified by two categorical variables. "Does the variant change conversion?" becomes "are rows and columns independent?" If the variant did nothing, both arms would convert at the pooled rate $440/2000 = 0.22$, so you would **expect** 220 conversions and 780 non-conversions per arm. The test measures how far the **observed** counts are from these.

~~~dg-fig 13.04-idea.svg
Observed counts (solid) against the counts expected if the variant did nothing (outlined). Each arm is 20 users away from 220 conversions; small cells contribute more per user of gap, because the gap is divided by the expected count.
~~~

Each cell is off by 20 users. The chi-square statistic squares each gap, divides by the expected count, and adds up: $1.82 + 1.82 + 0.51 + 0.51 = 4.66$, which gives $p = 0.031$.

## Step 1: the chi-square statistic

Both chi-square tests compare observed counts $O$ with the counts $E$ you would expect if the null were true:

$$\chi^2 = \sum_{\text{cells}} \frac{(O - E)^2}{E}.$$

- $O$ is the count you actually saw in a cell.
- $E$ is the count the null hypothesis predicts for that cell.
- $(O-E)^2/E$ is the cell's contribution: zero for a perfect match, larger the further off it is.

Why divide by $E$? A count that averages $E$ has a variance of about $E$ (Poisson behaviour), so $(O-E)/\sqrt{E}$ is roughly a z-score per cell and $\chi^2$ is a sum of squared z-scores. Hence its null distribution is the **chi-square distribution**, the distribution of a sum of squared standard normals.

The number of cells genuinely free to vary once the totals are fixed is the **degrees of freedom** (df).

~~~dg-fig 13.04-chi2.svg
The chi-square distribution for 1, 2 and 4 degrees of freedom. More df means more cells adding noise, so the curve shifts right and the 5% cut-off grows. The observed 4.66 on 1 df lies beyond 3.84; the shaded tail is the p-value.
~~~

## Step 2: goodness of fit (one variable)

Use this when you have one categorical variable with $k$ categories and a hypothesised share $\pi_j$ for each. The expected count in category $j$ is $E_j = n\pi_j$, and

$$\text{df} = k - 1,$$

because once $k-1$ counts are known the last is fixed by the total $n$. Subtract one more df for every parameter you estimated from the same data (for example, fitting a Poisson rate before checking a Poisson fit).

Its most common ML use is the **sample-ratio-mismatch (SRM) check**: with a planned 50/50 split, arm sizes should look like fair coin flips. A mismatch means assignment or logging is broken.

<div class="ex">
<div class="lab">Worked example</div>

**sample-ratio check on 100,000 users**

You intended a 50/50 split and got 50,600 users in A and 49,400 in B. Expected counts are 50,000 each.

Contributions: $(50{,}600 - 50{,}000)^2/50{,}000 = 360{,}000/50{,}000 = 7.2$, and the same 7.2 for B.

$\chi^2 = 7.2 + 7.2 = 14.4$ on $k - 1 = 1$ df. The 5% cut-off is 3.84; the p-value is 0.00015.

A 1.2-point imbalance looks harmless, but with 100,000 users it is far beyond chance. Find the bug (bots filtered in one arm, a redirect that drops users) before reading any metric.

</div>

## Step 3: independence (two variables)

Now there are two categorical variables in an $r \times c$ table (r rows, c columns). Under independence, the chance of landing in cell $(i, j)$ is the product of the row share and the column share, which gives

$$E_{ij} = \frac{(\text{row total}_i)(\text{column total}_j)}{n}, \qquad \text{df} = (r-1)(c-1).$$

<div class="ex">
<div class="lab">Worked example</div>

**does the new onboarding flow change conversion?**

Expected count for "A, converted": $1000 \cdot 440 / 2000 = 220$. The other cells: 780, 220, 780.

Gaps: every cell is off by 20. Contributions: $20^2/220 = 1.818$ twice and $20^2/780 = 0.513$ twice.

$\chi^2 = 3.636 + 1.026 = 4.66$ on $(2-1)(2-1) = 1$ df, so $p = 0.031$.

Cross-check with the two-proportion z-test from 13.03: $z = 0.04/\sqrt{0.22\cdot0.78\cdot(2/1000)} = 0.04/0.01853 = 2.159$, and $2.159^2 = 4.66$. For a $2\times2$ table the two tests are the same test.

</div>

This is no coincidence: a 1-df chi-square is one squared standard normal. The chi-square version generalises to bigger tables; the z version gives a signed difference and an interval, which is what you report.

## Step 4: when counts are small

The chi-square distribution is a large-sample approximation; the rule of thumb is expected counts of at least 5 in (nearly) all cells. With sparse cells, merge rare categories or use **Fisher's exact test**, which holds the margins fixed and computes the exact probability of every table at least as extreme as yours from the hypergeometric distribution. Take a table of model failures on two small slices: 3 of 10 failed on slice 1, 9 of 12 on slice 2. Expected counts are 4.5 to 6.5, two of them below 5. The uncorrected chi-square says $p = 0.035$; Fisher's exact test says $p = 0.084$. Same data, opposite verdict at $\alpha = 0.05$, and the exact test is the one to trust.

Note that `scipy.stats.chi2_contingency` applies **Yates' continuity correction** to $2\times2$ tables by default (4.43 instead of 4.66 on the onboarding table); pass `correction=False` to match the hand formula.

## Step 5: several group means with one-way ANOVA

Running all $\binom{k}{2}$ pairwise t-tests (10 for five groups) inflates false positives (lesson 13.06). **One-way ANOVA** (analysis of variance) asks one question first: are all group means equal?

The trick is to split the total variation in the data into two parts:

- **Between-group** variation, $\mathrm{SS_B}$: how far each group mean is from the grand mean, weighted by group size. Large when the groups really differ.
- **Within-group** variation, $\mathrm{SS_W}$: how far each point is from its own group mean. This is pure noise, whatever the null.

$$\mathrm{SS_B} = \sum_g n_g(\bar x_g - \bar x)^2, \qquad \mathrm{SS_W} = \sum_g\sum_{i\in g}(x_i - \bar x_g)^2, \qquad F = \frac{\mathrm{SS_B}/(k-1)}{\mathrm{SS_W}/(N-k)}.$$

Here $k$ is the number of groups, $n_g$ the size of group $g$, $\bar x_g$ its mean, $\bar x$ the grand mean and $N$ the total number of observations. Dividing each sum of squares by its df gives a **mean square** (MS), an average squared deviation.

Under $H_0$ both mean squares estimate the same noise variance, so $F$ hovers around 1; real group differences inflate only the numerator. The null distribution is the **F distribution** with $(k-1, N-k)$ df.

~~~dg-flow ANOVA in one picture: the same total variation, split into the part explained by group membership and the leftover noise.
Total variation -(split)-> Between-group SS -(divide by k-1)-> *F = MSB / MSW
Total variation -(split)-> Within-group SS -(divide by N-k)-> Noise estimate MSW
~~~

~~~dg-fig 13.04-anova.svg
Both panels have the same group means (5, 7, 9) and ten scores per template. Left: points sit tightly round their means, so the gaps dwarf the noise and F = 40. Right: three times the spread, so F falls by a factor of nine to 4.4.
~~~

Tripling the within-group standard deviation multiplies $\mathrm{MS_W}$ by 9: same means, $F$ from 40 to $40/9 = 4.4$.

<div class="ex">
<div class="lab">Worked example</div>

**three prompt templates, three scores each**

A = 4, 5, 6; B = 6, 7, 8; C = 8, 9, 10. Group means 5, 7, 9; grand mean 7.

Between: $\mathrm{SS_B} = 3(5-7)^2 + 3(7-7)^2 + 3(9-7)^2 = 3(4 + 0 + 4) = 24$ on $k - 1 = 2$ df, so $\mathrm{MS_B} = 12$.

Within: each group has deviations $(-1, 0, 1)$, so each contributes $1 + 0 + 1 = 2$. $\mathrm{SS_W} = 6$ on $N - k = 9 - 3 = 6$ df, so $\mathrm{MS_W} = 1$.

$F = 12/1 = 12$ on (2, 6) df, $p = 0.008$. Some template differs.

Follow-up with Tukey's HSD: only A versus C (a 4-point gap) is significant, $p = 0.006$; A versus B and B versus C (2-point gaps) both have $p = 0.109$. With three scores per group, a 2-point gap is not established on its own.

</div>

A significant $F$ says *some* mean differs, not which; use Tukey's HSD or pairwise tests with a multiple-testing correction to find out. Classical ANOVA assumes roughly normal residuals and equal variances; **Welch's ANOVA** drops the equal-variance assumption, and the **Kruskal-Wallis** test works on ranks and drops both. With exactly two groups, $F = t^2$ from the pooled two-sample t-test (for A versus B above, $t = -2.449$ and $F = 6.0$, same $p = 0.070$).

## Step 6: testing a correlation

Pearson's correlation $r$ (from 13.01) measures how tightly two numeric columns follow a straight line, from $-1$ to $+1$. To test whether the true correlation $\rho$ is zero, for $n$ pairs from a bivariate normal,

$$t = \frac{r\sqrt{n-2}}{\sqrt{1-r^2}} \sim t_{n-2}.$$

The df is $n - 2$ because a line has two parameters (intercept and slope); this is the same t-test as for the slope in simple linear regression.

**Number check.** With $r = 0.3$ and $n = 50$: $t = 0.3 \cdot \sqrt{48}/\sqrt{0.91} = 0.3 \cdot 6.928/0.954 = 2.18$ on 48 df, so $p = 0.034$.

For an interval, or to test $\rho = \rho_0 \ne 0$, the t formula does not help: when $\rho$ is far from zero, $r$ piles up against the wall at 1 and its distribution is lopsided. **Fisher's transform** $z = \operatorname{atanh}(r) = \tfrac12\ln\tfrac{1+r}{1-r}$ stretches the scale near $\pm1$ and makes the distribution approximately normal with standard error $1/\sqrt{n-3}$.

~~~dg-fig 13.04-fisherz.svg
20,000 simulated samples of 20 pairs with true correlation 0.8. Left: raw r is skewed with a long left tail (median 0.81, mean 0.79). Right: atanh(r) is symmetric and matches a normal curve whose standard error is 1/√17 ≈ 0.243 (simulated: 0.239).
~~~

<div class="ex">
<div class="lab">Worked example</div>

**a 95% interval for r = 0.3 with n = 50**

Transform: $z = \operatorname{atanh}(0.3) = 0.310$. Standard error: $1/\sqrt{47} = 0.146$.

Interval in z-space: $0.310 \pm 1.96 \cdot 0.146 = (0.024,\ 0.596)$.

Back-transform with tanh: $(\tanh 0.024,\ \tanh 0.596) = (0.024,\ 0.534)$.

The interval is asymmetric (0.28 below 0.3, 0.23 above), since there is less room towards 1. It excludes zero, matching $p = 0.034$, yet allows anything from negligible to fairly strong.

</div>

## Step 7: Pearson versus Spearman

Pearson's $r$ measures *linear* association and has a breakdown point of zero (lesson 13.01): one extreme point can create or destroy a correlation. **Spearman's** $\rho_s$ replaces each value by its rank and computes Pearson's $r$ on the ranks. It captures any **monotone** relation (always increasing or always decreasing, not necessarily a straight line) and resists outliers, because an extreme value is just "the largest rank".

~~~dg-fig 13.04-corr.svg
Left: on a noisy line the two measures agree. Middle: 29 unrelated points plus one far outlier give Pearson r = 0.73 but Spearman only 0.24. Right: an exponential-looking curve is perfectly ordered, so Spearman is 0.99 while Pearson is 0.84.
~~~

Neither implies causation, and with $n$ in the millions a correlation of 0.01 is significant and useless. Report $r$ with its interval and look at the scatter plot.

## Tests in code

```python
import numpy as np
from scipy import stats

# Counts: goodness of fit (sample-ratio check) and a 2x2 independence table
print(stats.chisquare([50_600, 49_400]))              # statistic 14.4, p 0.00015
table = np.array([[200, 800], [240, 760]])            # rows: A, B; cols: converted, not
res = stats.chi2_contingency(table, correction=False)
print(res.statistic, res.pvalue, res.dof)             # 4.662 0.0308 1
print(stats.chi2_contingency(table).statistic)        # 4.432 (Yates correction, the default)
print(stats.fisher_exact([[3, 7], [9, 3]]).pvalue)    # 0.0836: small table, use the exact test

# A 2x3 table: which cells drive the result? adjusted residuals
errs = np.array([[120, 60, 20], [90, 70, 40]])        # error types per model version
r = stats.chi2_contingency(errs)
print(round(r.statistic, 2), round(r.pvalue, 4))      # 11.72 0.0028
E = r.expected_freq; n = errs.sum()
row, col = errs.sum(1, keepdims=True) / n, errs.sum(0, keepdims=True) / n
print(np.round((errs - E) / np.sqrt(E * (1 - row) * (1 - col)), 2))
# [[ 3.   -1.07 -2.8 ]
#  [-3.    1.07  2.8 ]]

# Groups: one-way ANOVA, Welch ANOVA, rank version, and the follow-up
A, B, C = [4, 5, 6], [6, 7, 8], [8, 9, 10]
print(stats.f_oneway(A, B, C))                        # F 12.0, p 0.008
print(stats.f_oneway(A, B, C, equal_var=False))       # Welch: F 10.29, p 0.0265 (SciPy >= 1.15)
print(stats.kruskal(A, B, C).pvalue)                  # 0.038
print(stats.tukey_hsd(A, B, C).pvalue.round(3))       # A-C 0.006; A-B, B-C 0.109

# Correlation: Pearson with Fisher-z interval, Spearman on ranks
rng = np.random.default_rng(1)
x = rng.normal(size=100); y = 0.4 * x + rng.normal(size=100)
pr = stats.pearsonr(x, y)
print(round(pr.statistic, 3), round(pr.pvalue, 4))   # 0.302 0.0023
print(pr.confidence_interval(0.95))                   # (0.112, 0.470), Fisher z
print(stats.spearmanr(x, y))                          # rho 0.303, p 0.0022
```

## Reading and checking the result

A significant omnibus test (chi-square on a big table, ANOVA's $F$) is a pointer, not an answer.

- **Which cells or groups moved?** For tables, look at **adjusted residuals** $(O-E)/\sqrt{E(1-\text{row share})(1-\text{col share})}$, which are roughly standard normal under the null. In the error-type table above, version 2 has fewer errors of type 1 (residual $-3.0$) and more of type 3 ($+2.8$), while type 2 barely moved ($1.07$). For ANOVA, run Tukey's HSD.
- **How big is the effect?** For a $2\times2$ table report the difference in rates with its interval. For larger tables, **Cramér's V** $= \sqrt{\chi^2 / (n(\min(r,c)-1))}$ puts the strength on a 0-to-1 scale. For ANOVA, $\eta^2 = \mathrm{SS_B}/(\mathrm{SS_B}+\mathrm{SS_W})$ is the share of variation explained by group (0.8 in the template example). For correlation, $r$ itself and its interval.
- **Were the assumptions reasonable?** Expected counts of 5 or more; similar ANOVA group spreads (else Welch); a line-shaped cloud for Pearson.
- **Is the unit right?** Count whatever was randomised or sampled (users, patients), not events.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Counting events (clicks, errors, images) instead of independent units | tiny p-values from modest data; results flip between weeks | aggregate to the randomised unit, or use a cluster-aware or permutation test
| Chi-square on sparse tables | p-value disagrees with Fisher's exact test | merge rare categories, or use `fisher_exact`
| Forgetting Yates' default in SciPy | your hand $\chi^2$ and SciPy's differ on a $2\times2$ table | pass `correction=False` if you want the z-test equivalent
| Many pairwise t-tests instead of ANOVA | some "significant" pair almost guaranteed | ANOVA, then Tukey or a corrected follow-up
| Classical ANOVA with very unequal variances and sizes | false positives or lost power | Welch ANOVA, or Kruskal-Wallis
| Pearson on curved or outlier-heavy data | r driven by one point or understating a monotone link | plot first; use Spearman

The first row deserves a picture. Below, both arms are identical (an **A/A test**) with 200 users per arm, and each user has their own click rate, so one user's clicks are correlated.

~~~dg-fig 13.04-cluster.svg
Simulated A/A tests (2,000 per point, no real effect). Treating events as independent gives a false-positive rate of 9% at 5 events per user, 23% at 20 and 39% at 50. Testing one number per user stays at about 5% throughout.
~~~

The test thinks it has $200 \times 50 = 10{,}000$ independent observations per arm, but holds closer to 200 users' worth of information, so its standard error is far too small.

## Going deeper

**Where the chi-square comes from.** Pearson's statistic is a quadratic approximation to the **likelihood-ratio** (G) statistic $G = 2\sum O\ln(O/E)$; both converge to the same $\chi^2$ distribution as counts grow, and $G$ is available as `chi2_contingency(..., lambda_="log-likelihood")`. The df count is "free parameters under the alternative minus free parameters under the null": an $r\times c$ table has $rc - 1$ free cell probabilities, independence needs only $(r-1)+(c-1)$, and the difference is $(r-1)(c-1)$.

**Why Fisher's test is "exact" but conservative.** Conditioning on both margins removes the unknown rate, but the table takes only a few discrete values, so the real false-positive rate is at or below $\alpha$, often well below. Mid-p values or Barnard's test (`scipy.stats.barnard_exact`) are less conservative alternatives.

**ANOVA is a linear model.** One-way ANOVA is linear regression on one-hot group indicators, and $F$ compares the residual sum of squares of that model with the intercept-only model. This extends to two-way ANOVA, ANCOVA and mixed models with a random effect per user, a principled fix for clustering.

**Fisher's z derivation in one line.** By the delta method, $\operatorname{Var}(r) \approx (1-\rho^2)^2/n$, which depends on $\rho$. The function whose derivative is $1/(1-\rho^2)$ is $\operatorname{atanh}$, so $\operatorname{atanh}(r)$ has approximately constant variance, refined to $1/(n-3)$. This is a **variance-stabilising transform**, the same idea as taking square roots of Poisson counts.

## Key terms

| Term | Meaning
| --- | --- |
| Contingency table | Grid of counts cross-classifying units by two categorical variables
| Expected count $E$ | The count a cell would have on average if the null were true
| Chi-square statistic | $\sum (O-E)^2/E$, a sum of squared count-level z-scores
| Degrees of freedom | How many cells (or values) can vary once the totals are fixed
| Fisher's exact test | Exact test for small tables using the hypergeometric distribution
| One-way ANOVA | Tests equality of $k$ group means with $F = \mathrm{MS_B}/\mathrm{MS_W}$
| Tukey's HSD | Follow-up that compares every pair of groups while controlling false positives
| Fisher's z | $\operatorname{atanh}(r)$, approximately normal with SE $1/\sqrt{n-3}$; used for intervals
| Spearman's $\rho_s$ | Pearson correlation computed on ranks; measures monotone association
| Unit of analysis | The independent thing that was sampled or randomised (user, patient), not events

<div class="take">
<div class="lab">Key takeaway</div>

use chi-square (or Fisher's exact test when expected counts are small) for counts, one ANOVA instead of many t-tests for several group means, and the t or Fisher-z test for correlations, always on independent units, and treat a significant omnibus result as a pointer to follow-up comparisons and effect sizes, not an answer.

</div>
