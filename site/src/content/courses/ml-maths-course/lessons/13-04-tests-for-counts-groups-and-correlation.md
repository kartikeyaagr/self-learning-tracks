---
id: "13.04"
course: "ml-maths-course"
module: "13"
slug: "tests-for-counts-groups-and-correlation"
title: "Tests for counts, groups and correlation"
keyIdeas: ["Chi-square tests compare observed counts with expected counts: $\\chi^2 = \\sum (O-E)^2/E$, with $k-1$ degrees of freedom for goodness of fit and $(r-1)(c-1)$ for an $r\\times c$ independence table.", "For a $2\\times2$ table the uncorrected chi-square statistic equals the square of the two-proportion z statistic; when expected counts fall below about 5, use Fisher's exact test.", "One-way ANOVA tests whether $k$ group means are all equal with $F = \\mathrm{MS_{between}}/\\mathrm{MS_{within}}$; a significant F says some mean differs, not which.", "A Pearson correlation is tested with $t = r\\sqrt{n-2}/\\sqrt{1-r^2}$ on $n-2$ degrees of freedom; Fisher's $z = \\operatorname{atanh} r$ gives intervals. Spearman's rank correlation handles monotone, non-linear or outlier-prone data.", "All of these assume independent observations; counting events instead of units (clicks instead of users) silently breaks them."]
question: "You compare error-type distributions (5 error categories) between two model versions using every logged error: 12,000 errors from version 1 and 11,500 from version 2, collected from 300 users. The chi-square test of independence gives p = 1e-9. What would you check before concluding the versions fail differently?"
answer: "First, independence: the 23,500 errors come from only 300 users, and a few heavy users with their own peculiar inputs can generate thousands of correlated errors, so the chi-square treats clustered events as independent and the p-value is far too small. Re-run the analysis with the user as the unit, for example per-user error-type proportions compared with a permutation test that shuffles version labels across users, or a user-level bootstrap. Second, confounding: did the same users and traffic mix see both versions, or did version 2 roll out to different segments or later dates? Third, effect size: with 23,500 counts tiny shifts are significant, so inspect standardised residuals $(O-E)/\\sqrt E$ per cell to see which categories moved and by how much, and judge whether that matters. Finally, check expected counts in rare categories."
sources: [{"title": "SciPy: scipy.stats.chi2_contingency", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html", "note": "Expected frequencies, degrees of freedom and the Yates correction default."}, {"title": "NIST/SEMATECH e-Handbook: Chi-Square Goodness-of-Fit Test", "url": "https://www.itl.nist.gov/div898/handbook/eda/section3/eda35f.htm", "note": "Binning, expected counts and degrees of freedom."}, {"title": "SciPy: scipy.stats.f_oneway", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html", "note": "One-way ANOVA, its assumptions and the alternatives SciPy points to."}, {"title": "SciPy: scipy.stats.pearsonr", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html", "note": "The t-based test and Fisher-z confidence interval for r."}]
practice: ["chi-square-goodness-of-fit", "independence-testing", "anova-f-statistic", "correlation-significance-test"]
tags: ["chi-square", "ANOVA", "correlation"]
tools: ["SciPy"]
---

## Beyond one mean

Lesson 13.03 compared means. Much ML data is categorical (did the user convert, which error class did the model produce) or involves more than two groups (four prompt templates, five data sources). Three families of tests cover these cases, and all of them reuse the pipeline from 13.03: a statistic, a null distribution, a tail area.

| Question | Data | Test | Null distribution
| --- | --- | --- | --- |
| Do counts match a target distribution? | one categorical variable | chi-square goodness of fit | $\chi^2_{k-1}$
| Are two categorical variables related? | contingency table | chi-square independence, Fisher exact | $\chi^2_{(r-1)(c-1)}$
| Do several group means differ? | numeric outcome, k groups | one-way ANOVA (Welch ANOVA if variances differ) | $F_{k-1,\,N-k}$
| Are two numeric variables linearly related? | paired numeric columns | Pearson t-test, Fisher z | $t_{n-2}$
| Monotone relation, outliers, ranks? | paired columns | Spearman rank correlation | approx. $t_{n-2}$ or permutation

## Chi-square tests for counts

Both chi-square tests compare observed counts $O$ to the counts $E$ you would expect under the null:

$$\chi^2 = \sum_{\text{cells}} \frac{(O - E)^2}{E}.$$

- **Goodness of fit**: one variable with $k$ categories and hypothesised proportions $\pi_j$, so $E_j = n\pi_j$ and $k-1$ degrees of freedom (subtract one more for each parameter estimated from the data). A sample-ratio-mismatch check in an A/B test is exactly this with $k=2$ and $\pi = (0.5, 0.5)$.
- **Independence**: an $r\times c$ table of counts. Under independence $E_{ij} = (\text{row}_i\cdot\text{col}_j)/n$, with $(r-1)(c-1)$ degrees of freedom.

The $\chi^2$ approximation is a large-sample result. The usual rule is expected counts of at least 5 in (nearly) all cells; with sparse tables merge categories or use **Fisher's exact test**, which computes the probability of the table directly from the hypergeometric distribution. Note that `scipy.stats.chi2_contingency` applies Yates' continuity correction to $2\times2$ tables by default, which makes it more conservative; pass `correction=False` to match the hand formula.

<div class="ex">
<div class="lab">Worked example</div>

**does the new onboarding flow change conversion?**

Variant A: 200 of 1,000 users converted; variant B: 240 of 1,000. The pooled rate is 0.22, so expected counts are 220 converted and 780 not in each arm.

$\chi^2 = 2\cdot\frac{20^2}{220} + 2\cdot\frac{20^2}{780} = 3.636 + 1.026 = 4.66$ on 1 degree of freedom, $p = 0.031$.

The two-proportion z-test gives $z = 0.04/\sqrt{0.22\cdot0.78\cdot(2/1000)} = 2.159$, and $2.159^2 = 4.66$: for a $2\times2$ table the two tests are identical.

</div>

## One-way ANOVA

With $k$ groups, running all $\binom{k}{2}$ t-tests inflates false positives (lesson 13.06). ANOVA asks a single question: are all group means equal? It splits the total variation into variation between group means and variation within groups:

$$F = \frac{\mathrm{SS_B}/(k-1)}{\mathrm{SS_W}/(N-k)}, \quad \mathrm{SS_B} = \sum_g n_g(\bar x_g - \bar x)^2, \quad \mathrm{SS_W} = \sum_g\sum_{i\in g}(x_i - \bar x_g)^2.$$

Under $H_0$ both mean squares estimate the noise variance, so $F\approx1$; real differences inflate the numerator. A significant $F$ tells you that some mean differs. Follow-up comparisons (Tukey's HSD, or pairwise tests with a multiple-testing correction) say which. Classical ANOVA assumes roughly normal residuals and equal variances; Welch's ANOVA drops the second assumption, and the Kruskal-Wallis rank test drops both. With two groups, $F = t^2$ from the pooled two-sample t-test.

~~~dg-flow ANOVA in one picture: the same total variation, split into the part explained by group membership and the leftover noise.
Total variation -(split)-> Between-group SS -(divide by k-1)-> *F = MSB / MSW
Total variation -(split)-> Within-group SS -(divide by N-k)-> Noise estimate MSW
~~~

<div class="ex">
<div class="lab">Worked example</div>

**three prompt templates, three scores each**

A = 4, 5, 6; B = 6, 7, 8; C = 8, 9, 10. Group means 5, 7, 9; grand mean 7.

$\mathrm{SS_B} = 3(4 + 0 + 4) = 24$ on 2 df, so $\mathrm{MS_B} = 12$. $\mathrm{SS_W} = 2 + 2 + 2 = 6$ on 6 df, so $\mathrm{MS_W} = 1$.

$F = 12$ on (2, 6) df, $p = 0.008$. The templates differ; pairwise follow-ups decide whether A versus B (a 2-point gap) holds up on its own.

</div>

## Testing a correlation

For $n$ pairs from a bivariate normal with true correlation zero,

$$t = \frac{r\sqrt{n-2}}{\sqrt{1-r^2}} \sim t_{n-2}.$$

With $r = 0.3$ and $n = 50$, $t = 2.18$ and $p = 0.034$. For a confidence interval, or to test $\rho = \rho_0 \ne 0$, use **Fisher's transform** $z = \operatorname{atanh}(r)$, which is approximately normal with SE $1/\sqrt{n-3}$. Build the interval in $z$-space and transform back with $\tanh$; the result is asymmetric, as it should be near $\pm1$.

Pearson's $r$ measures linear association and has a breakdown point of zero (lesson 13.01): one extreme point can create or destroy a correlation. **Spearman's** $\rho_s$ is Pearson's $r$ on ranks, so it captures any monotone relation and resists outliers. Neither says anything about causation, and with $n$ in the millions a correlation of 0.01 is significant and useless.

## The assumption that breaks everything: independence

Every test above assumes the observations are independent. A table of 50,000 clicks from 2,000 users, or ANOVA over images of which many come from the same patient, overstates $n$ and produces tiny p-values. Aggregate to the unit that was randomised or sampled before testing, exactly as in lesson 13.02.

<div class="take">
<div class="lab">Key takeaway</div>

use chi-square (or Fisher's exact test when counts are small) for counts, one ANOVA instead of many t-tests for several group means, and the t or Fisher-z test for correlations, always on independent units, and treat a significant omnibus result as a pointer to follow-up comparisons, not an answer.

</div>
