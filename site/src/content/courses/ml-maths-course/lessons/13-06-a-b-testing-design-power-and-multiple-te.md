---
id: "13.06"
course: "ml-maths-course"
module: "13"
slug: "a-b-testing-design-power-and-multiple-testing"
title: "A/B testing: design, power and multiple testing"
keyIdeas: ["Fix the randomisation unit, one primary metric, guardrails, $\\alpha$, power and the minimum detectable effect **before** the test starts; analyse at the unit you randomised.", "Sample size per arm is $n \\approx (z_{1-\\alpha/2}+z_{1-\\beta})^2\\,(\\sigma_A^2+\\sigma_B^2)/\\delta^2$, about $16\\sigma^2/\\delta^2$ at 5% and 80% power: halving the detectable effect quadruples traffic.", "Peeking at a fixed-horizon test and stopping at the first $p<0.05$ inflates false positives (about 19% with 10 looks); use a fixed horizon or a sequential method.", "Many metrics or segments need a correction: Bonferroni controls the chance of any false positive, Benjamini-Hochberg controls the fraction of false discoveries and keeps more power.", "Check sample-ratio mismatch with a chi-square test first; variance reduction such as CUPED cuts the needed sample by a factor of $1-\\rho^2$."]
question: "A test with 20 secondary metrics runs for two weeks. The primary metric is flat (p = 0.4), but 'time on page for Android users in Germany' is up 12% with p = 0.01. The PM wants to ship. What is your analysis?"
answer: "The headline result is the primary metric, and it shows no detectable effect. The German Android finding is one of a very large number of metric-by-segment combinations that could have been examined (20 metrics times many segments), so at least one p near 0.01 is expected by chance; with, say, 200 implicit comparisons, a Bonferroni threshold would be 0.00025, and Benjamini-Hochberg would almost certainly not reject a lone 0.01. The segment was also chosen after seeing the data, which invalidates the nominal p-value. The 12% lift is likely inflated by the same selection. Correct move: treat it as a hypothesis, not a result. Do not ship on it; if the segment matters, run a new test pre-registered on that segment and metric, sized by a power calculation, and confirm SRM and guardrails as usual."
sources: [{"title": "Kohavi et al.: Controlled experiments on the web: survey and practical guide", "url": "https://link.springer.com/article/10.1007/s10618-008-0114-1", "note": "Randomisation units, the overall evaluation criterion, power and common pitfalls."}, {"title": "Evan Miller: How Not To Run an A/B Test", "url": "https://www.evanmiller.org/how-not-to-run-an-ab-test.html", "note": "A short, concrete account of why peeking inflates false positives."}, {"title": "Johari et al.: Always Valid Inference", "url": "https://arxiv.org/abs/1512.04922", "note": "Always-valid p-values (mixture SPRT) for continuously monitored A/B tests."}, {"title": "Deng et al.: Improving the Sensitivity of Online Controlled Experiments (CUPED)", "url": "https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf", "note": "Variance reduction with pre-experiment data."}]
practice: ["b-test-significance", "p-values-and-significance", "msprt-always-valid-pvalues"]
tags: ["A/B testing", "power analysis", "multiple testing"]
tools: ["SciPy"]
---

## Experiments put the module together

An online A/B test randomly splits units (users, sessions, devices) between a control and a treatment and compares a metric. Randomisation is what makes the comparison causal: any difference beyond chance is due to the change. Every earlier lesson appears here: robust metrics (13.01), standard errors with the right unit (13.02), z and t tests (13.03), chi-square checks (13.04) and the bootstrap for awkward metrics (13.05). This lesson covers the statistics. The Machine Learning course's production module covers shipping models through canaries and sequential tests in more depth.

~~~dg-flow An experiment's decisions are made before data arrive; analysis only executes the plan.
Hypothesis -(choose)-> Unit, metric, MDE -(power calc)-> Sample size -(run, no peeking)-> SRM check -(test)-> *Effect + CI -(decide)-> Ship or not
~~~

## Design decisions

- **Randomisation unit.** Usually the user, so one person sees one experience. Analyse at that unit: a per-page-view analysis of a user-randomised test understates the SE (13.02).
- **Primary metric.** One metric decides (an overall evaluation criterion such as conversion or revenue per user). Guardrail metrics (latency, error rate, unsubscribes) must not degrade.
- **Minimum detectable effect (MDE).** The smallest effect worth detecting, set by the business, not by what the data happen to show.
- **$\alpha$ and power.** Commonly $\alpha = 0.05$ two-sided and power $1-\beta = 0.8$.
- **Duration.** At least one full weekly cycle, so weekday and weekend users are both represented, and long enough to reach the sample size.

## Power and sample size

To detect a difference $\delta$ between two means with noise $\sigma_A$, $\sigma_B$, the per-arm sample size is

$$n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2\,(\sigma_A^2 + \sigma_B^2)}{\delta^2}.$$

With $\alpha = 0.05$ and 80% power, $(1.96 + 0.84)^2 \approx 7.85$, so with equal variances $n \approx 16\sigma^2/\delta^2$. For a conversion rate, $\sigma^2 = p(1-p)$.

<div class="ex">
<div class="lab">Worked example</div>

**sizing a checkout test**

Baseline conversion 5.0%; the MDE is a 10% relative lift, to 5.5%.

$n = 7.85\cdot(0.05\cdot0.95 + 0.055\cdot0.945)/0.005^2 = 7.85\cdot0.0995/0.000025 \approx 31{,}200$ users per arm, 62,400 in total.

For a 5% relative lift (to 5.25%) the required $n$ roughly quadruples, to about 122,000 per arm. With 4,000 eligible users a day, that is the difference between about 2 weeks and 2 months.

Running the smaller test and "seeing what happens" for a 5% lift gives power of about 29%: most real 5% wins would be missed, and the ones that do reach significance would overstate the lift.

</div>

## Peeking and sequential testing

The p-value from 13.03 assumes one analysis at a pre-set $n$. Checking every day and stopping at the first $p < 0.05$ gives the noise many chances to cross the line. In simulation, 10 equally spaced looks at an A/A test (no true effect) produce a "significant" result about 19% of the time instead of 5%; continuous monitoring pushes it towards certainty. Two honest alternatives:

- **Fixed horizon**: compute $n$, wait, analyse once. Dashboards can show data, but no decision is made until the end.
- **Sequential designs**: group-sequential boundaries (O'Brien-Fleming spends little $\alpha$ early) or always-valid p-values such as the mixture SPRT, which stay correct under continuous monitoring at some cost in power.

## Multiple testing

Twenty metrics, or one metric in twenty segments, at $\alpha = 0.05$ each: if nothing changed, the chance of at least one false positive is $1 - 0.95^{20} = 64\%$.

- **Bonferroni** controls the family-wise error rate (FWER, the probability of any false positive) by testing each of $m$ hypotheses at $\alpha/m$. Simple and valid under any dependence, but conservative for large $m$.
- **Benjamini-Hochberg (BH)** controls the false discovery rate (FDR, the expected fraction of rejections that are false). Sort the p-values $p_{(1)} \le \dots \le p_{(m)}$, find the largest $k$ with $p_{(k)} \le \frac{k}{m}q$, and reject hypotheses $1..k$.

| Sorted p-value | 0.001 | 0.012 | 0.021 | 0.040 | 0.300
| --- | --- | --- | --- | --- | --- |
| BH threshold $kq/m$ ($q = 0.05$, $m = 5$) | 0.010 | 0.020 | 0.030 | 0.040 | 0.050
| BH rejects | yes | yes | yes | yes | no
| Bonferroni ($p \le 0.01$) rejects | yes | no | no | no | no

Use FWER control for the primary decision and FDR when screening many secondary metrics or segments. Better still, name one primary metric in advance so that only one test decides.

## Sanity checks and variance reduction

- **Sample-ratio mismatch (SRM).** A 50/50 test that ends with 50,000 versus 49,000 users gives $\chi^2 = 2\cdot500^2/49{,}500 = 10.1$, $p = 0.0015$ (13.04). An SRM means assignment or logging is broken (bots filtered in one arm, a crash before logging), and the result should not be trusted until it is explained.
- **A/A tests** check the whole pipeline: over many A/A runs, about 5% should be significant.
- **CUPED** regresses the metric on a pre-experiment covariate (the same metric last month) and analyses the residual $Y - \theta(X - \bar X)$. Variance is multiplied by $1-\rho^2$; with $\rho = 0.7$ it falls to 51% of the original, which roughly halves the sample size required.
- **Novelty and interference.** Early lifts can fade; users in a marketplace or social network affect each other, which breaks the independence between arms and may need cluster-level randomisation.

```
from scipy import stats
z_a, z_b = stats.norm.ppf(0.975), stats.norm.ppf(0.80)
p1, p2 = 0.05, 0.055
n = (z_a + z_b) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2
p_adj = stats.false_discovery_control([0.001, 0.012, 0.021, 0.04, 0.3])  # BH-adjusted
```

<div class="take">
<div class="lab">Key takeaway</div>

an A/B test is decided at design time: fix the unit, one primary metric, the MDE and the sample size from a power calculation, then analyse once (or with a proper sequential method), check SRM first, and correct for every extra metric or segment you look at.

</div>
