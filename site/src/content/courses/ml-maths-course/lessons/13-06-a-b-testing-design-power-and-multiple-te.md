---
id: "13.06"
course: "ml-maths-course"
module: "13"
slug: "a-b-testing-design-power-and-multiple-testing"
title: "A/B testing: design, power and multiple testing"
keyIdeas: ["Fix the randomisation unit, one primary metric, guardrails, $\\alpha$, power and the **minimum detectable effect** before the test starts, and analyse at the unit you randomised.", "Sample size per arm is $n \\approx (z_{1-\\alpha/2}+z_{1-\\beta})^2\\,(\\sigma_A^2+\\sigma_B^2)/\\delta^2$, about $16\\sigma^2/\\delta^2$ at 5% and 80% power: halving the detectable effect quadruples the traffic.", "Peeking at a fixed-horizon test and stopping at the first $p<0.05$ inflates false positives (about 19% with 10 looks); use a fixed horizon or a planned sequential method.", "Many metrics or segments need a correction: Bonferroni controls the chance of any false positive, Benjamini-Hochberg controls the fraction of false discoveries and keeps more power.", "Check sample-ratio mismatch with a chi-square test before reading any metric; CUPED cuts variance (and the needed sample) by a factor of $1-\\rho^2$.", "Underpowered tests miss most real effects and exaggerate the ones they catch (the winner's curse), so read the confidence interval against the MDE, not just $p$ against 0.05."]
question: "A test with 20 secondary metrics runs for two weeks. The primary metric is flat (p = 0.4), but 'time on page for Android users in Germany' is up 12% with p = 0.01. The PM wants to ship. What is your analysis?"
answer: "The headline result is the primary metric, and it shows no detectable effect; check its confidence interval to say whether it is a confident null (narrow, around zero) or just underpowered (wide). The German Android finding is one of a very large number of metric-by-segment combinations that could have been examined (20 metrics times many segments), so at least one p near 0.01 is expected by chance: with 20 independent null metrics alone, the chance of some p below 0.05 is $1-0.95^{20} = 64\\%$. With, say, 200 implicit comparisons, a Bonferroni threshold would be $0.05/200 = 0.00025$, and Benjamini-Hochberg would almost certainly not reject a lone 0.01 (its smallest threshold is $q/m = 0.00025$ too, unless many other p-values are also small). The segment was also chosen after seeing the data, which invalidates the nominal p-value entirely, and segment-level samples are small, so the 12% lift is likely inflated by the winner's curse. Before anything else, confirm there is no sample-ratio mismatch overall or within the segment. Correct move: treat it as a hypothesis, not a result. Do not ship on it; if the segment matters, run a new test pre-registered on that segment and metric, sized by a power calculation for a realistic MDE (not the 12% seen here), and check SRM and guardrails as usual."
sources: [{"title": "Kohavi et al.: Controlled experiments on the web: survey and practical guide", "url": "https://link.springer.com/article/10.1007/s10618-008-0114-1", "note": "Randomisation units, the overall evaluation criterion, power and common pitfalls."}, {"title": "Evan Miller: How Not To Run an A/B Test", "url": "https://www.evanmiller.org/how-not-to-run-an-ab-test.html", "note": "A short, concrete account of why peeking inflates false positives."}, {"title": "Johari et al.: Always Valid Inference", "url": "https://arxiv.org/abs/1512.04922", "note": "Always-valid p-values (mixture SPRT) for continuously monitored A/B tests."}, {"title": "Deng et al.: Improving the Sensitivity of Online Controlled Experiments (CUPED)", "url": "https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf", "note": "Variance reduction with pre-experiment data."}]
practice: ["b-test-significance", "p-values-and-significance", "msprt-always-valid-pvalues"]
tags: ["A/B testing", "power analysis", "multiple testing"]
tools: ["SciPy"]
---

## What you will learn

- What an A/B test is and why random assignment lets you read a difference as caused by the change.
- The design decisions that must be fixed before the first user arrives: randomisation unit, primary metric, guardrails, minimum detectable effect, $\alpha$ and power.
- How to size a test by hand with the power formula, and why halving the effect you want to detect quadruples the traffic you need.
- How to analyse once and honestly: sample-ratio check first, then a two-proportion test and a confidence interval.
- The three classic traps: peeking, many metrics or segments, and underpowered tests that exaggerate their wins, plus CUPED, which shrinks the noise.

*Before you start:* standard errors (13.02), z tests, p-values and power (13.03), the chi-square goodness-of-fit test (13.04) and the bootstrap (13.05). The code uses NumPy and SciPy.

## Experiments put the module together

Lesson 13.05 showed how to test almost anything by resampling. This lesson puts every tool of the module to work on the most common statistical job in industry: deciding whether a change to a product, a page or a model actually helps.

An online **A/B test** (a randomised controlled experiment) splits **units** (users, sessions, devices) at random into a **control** arm A, which sees the current version, and a **treatment** arm B, which sees the change. Then it compares a metric between the arms. Because a coin flip decided who got what, the groups differ only by chance and by the change, so any difference larger than chance can explain is caused by the change. Observational data cannot give you that causal reading. The Machine Learning course's production module covers shipping models through canaries and sequential tests in more depth.

## The idea in one picture

A shop wants to know if a new one-page checkout raises the share of visitors who buy. Today 5.0% convert. The team decides that a lift to 5.5% (a 10% relative lift) is worth shipping, and they plan 31,250 users per arm. Two weeks later:

| Arm | Users | Purchases | Conversion
| --- | --- | --- | --- |
| A (old checkout) | 31,250 | 1,571 | 5.03%
| B (new checkout) | 31,250 | 1,734 | 5.55%

The difference is $+0.52$ percentage points. Is that the new checkout, or luck? The figure answers by imagining the test run many times.

~~~dg-fig 13.06-idea.svg
With 31,250 users per arm the estimated difference wobbles by about 0.18 points. If nothing changed (blue) it lands beyond ±0.34 only 5% of the time; if the true lift is the planned 0.5 points (green) it clears the bar 81% of the time. The observed +0.52 is well past the bar.
~~~

Read the picture in three parts:

- The **blue curve** is what the difference looks like across repeated tests when the change does nothing. Its spread is the standard error, about 0.18 points here.
- The **dashed bars** at $\pm 0.34$ points are the **significance bar**: a difference that far from zero happens by chance only 5% of the time. That 5% is $\alpha$, the false positive rate you accept.
- The **green curve** is what you would see if the true lift were exactly the 0.5 points you care about. The shaded share of it past the bar is the **power**: the chance the test detects an effect of that size. Here it is 81%.

Design means choosing the sample size so that these two curves overlap only a little.

## Step 1: fix the design before data arrive

Every decision that could be bent by looking at results is made in advance and written down.

~~~dg-flow An experiment's decisions are made before data arrive; analysis only executes the plan.
Hypothesis -(choose)-> Unit, metric, MDE -(power calc)-> Sample size -(run, no peeking)-> SRM check -(test)-> *Effect + CI -(decide)-> Ship or not
~~~

- **Randomisation unit.** Usually the user, so one person sees one experience. Analyse at the same unit: treating a heavy user's 40 page views as 40 independent observations understates the standard error (13.02).
- **Primary metric.** One metric decides (the overall evaluation criterion), such as conversion or revenue per user. Naming it in advance is the cheapest defence against multiple testing (Step 5).
- **Guardrail metrics.** Things that must not get worse (latency, error rate, unsubscribes). They veto a harmful win.
- **Minimum detectable effect (MDE).** The smallest true effect worth detecting, set by the business, never by what the data happen to show.
- **$\alpha$ and power.** Commonly $\alpha = 0.05$ two-sided and power $1-\beta = 0.8$, where $\beta$ is the chance of missing a real effect of size MDE.
- **Duration.** Whole weeks (weekday and weekend users differ), and long enough to reach the planned sample size. Early lifts can also be novelty that fades.

## Step 2: compute the sample size

In words: the test needs enough users that the MDE sits about 2.8 standard errors from zero. Then 1.96 of those standard errors are used by the significance bar, and the remaining 0.84 put 80% of the green curve past it.

The standard error of a difference of two means with $n$ users per arm is $\sqrt{(\sigma_A^2 + \sigma_B^2)/n}$. Setting $\delta = (z_{1-\alpha/2} + z_{1-\beta}) \times \text{SE}$ and solving for $n$ gives the per-arm sample size

$$n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2\,(\sigma_A^2 + \sigma_B^2)}{\delta^2}.$$

The symbols:

- $\delta$ is the MDE in the metric's own units (here 0.005, half a percentage point).
- $\sigma_A^2, \sigma_B^2$ are the per-user variances of the metric in each arm. For a 0/1 conversion metric with rate $p$, $\sigma^2 = p(1-p)$.
- $z_{1-\alpha/2} = 1.96$ for $\alpha = 0.05$ two-sided, and $z_{1-\beta} = 0.84$ for 80% power.

With these defaults $(1.96 + 0.84)^2 \approx 7.85$, so with equal variances $n \approx 7.85 \times 2\sigma^2/\delta^2 \approx 16\sigma^2/\delta^2$. The quick check to remember: **halving $\delta$ quadruples $n$**, because $\delta$ is squared in the denominator.

<div class="ex">
<div class="lab">Worked example</div>

**sizing the checkout test**

Baseline conversion 5.0%; the MDE is a 10% relative lift, to 5.5%, so $\delta = 0.005$.

Variances: $0.05 \times 0.95 = 0.0475$ and $0.055 \times 0.945 = 0.0520$; their sum is $0.0995$.

$n = 7.85 \times 0.0995 / 0.005^2 = 0.781 / 0.000025 \approx 31{,}200$ users per arm, about 62,500 in total.

With 4,000 eligible users a day, that is about 16 days: two full weeks, which also covers two weekly cycles.

For a 5% relative lift (to 5.25%, $\delta = 0.0025$) the variance sum is $0.0972$ and $n = 7.85 \times 0.0972 / 0.0025^2 \approx 122{,}000$ per arm: about 61 days of traffic.

Running the 31,200-per-arm test anyway for a true 5% lift gives power of only about 29%: most real 5% wins would be missed.

</div>

~~~dg-fig 13.06-samplesize.svg
Users needed per arm against the relative MDE, on a log scale. The curve is steep: going from a 10% to a 5% detectable lift moves you from two weeks to two months. A higher baseline rate needs far fewer users for the same relative lift.
~~~

## Step 3: check the plumbing, then analyse once

**First, the sample-ratio check.** A **sample-ratio mismatch (SRM)** is an arm-size gap too big for chance, tested with the chi-square goodness-of-fit test (13.04). For example, 50,000 versus 49,000 users gives $\chi^2 = 2 \times 500^2 / 49{,}500 = 10.1$ and $p = 0.0015$. An SRM means assignment or logging is broken (bots filtered in one arm, a crash before logging); do not trust the result until it is explained. Our checkout test has exactly 31,250 per arm, so it passes.

**Then the test.** For conversion rates use the two-proportion z test from 13.03. Under "no difference" both arms share one rate, estimated by pooling, so the z statistic is

$$z = \frac{\hat p_B - \hat p_A}{\sqrt{\hat p(1-\hat p)\,(1/n_A + 1/n_B)}},$$

where $\hat p_A, \hat p_B$ are the observed rates and $\hat p$ is the pooled rate. For the confidence interval use the unpooled standard error $\sqrt{\hat p_A(1-\hat p_A)/n_A + \hat p_B(1-\hat p_B)/n_B}$.

<div class="ex">
<div class="lab">Worked example</div>

**analysing the checkout test**

Rates: $\hat p_A = 1571/31250 = 0.05027$ and $\hat p_B = 1734/31250 = 0.05549$. Difference $0.00522$.

Pooled rate: $\hat p = 3305/62500 = 0.05288$. Pooled SE: $\sqrt{0.05288 \times 0.94712 \times 2/31250} = \sqrt{3.205 \times 10^{-6}} = 0.00179$.

$z = 0.00522/0.00179 = 2.91$, two-sided $p = 2 \times (1 - \Phi(2.91)) = 0.0036$.

Unpooled SE is also $0.00179$ to three digits, so the 95% CI is $0.00522 \pm 1.96 \times 0.00179 = 0.00522 \pm 0.00351$, which is $[0.17, 0.87]$ percentage points.

Relative lift: $0.52/5.03 = 10.4\%$ (interval roughly 3% to 17%). Significant, and the interval is entirely above zero.

</div>

For continuous metrics use Welch's t test (13.03); for heavy-tailed or ratio metrics, a user-level bootstrap (13.05).

## Step 4: do not peek, or peek properly

The p-value from 13.03 assumes one analysis at a sample size fixed in advance. Dashboards update daily, and the temptation is to stop the moment $p < 0.05$. Each look is a fresh chance for noise to cross the bar.

~~~dg-fig 13.06-peeking.svg
Left: running z statistics for four A/A tests with no real effect. Two wander past ±1.96 mid-way and would have been "winners" if stopped there, yet end inside the bar. Right: in 20,000 simulated A/A tests the false positive rate grows from 5% with one look to about 19% with 10 looks and 38% with 100.
~~~

In an A/A test both arms are identical, so every "significant" result is false. Continuous monitoring pushes the false positive rate towards certainty as the test runs on. Two honest alternatives:

- **Fixed horizon.** Compute $n$, wait, analyse once. Dashboards can show data for bug-spotting (SRM, crashes), but no ship decision is made until the end.
- **Sequential designs.** Plan the looks and widen the bar at early looks. A **group-sequential** design with O'Brien-Fleming boundaries (the curved bar in the left panel) uses a very strict bar early and almost 1.96 at the end, so it costs little power. Always-valid p-values, such as the mixture sequential probability ratio test (mSPRT), stay correct under continuous monitoring at some cost in power.

## Step 5: correct for many metrics and segments

A report shows twenty metrics, and someone asks for splits by country and platform. Each extra comparison at $\alpha = 0.05$ is another 5% lottery ticket. If nothing changed and the twenty tests were independent, the chance of at least one false positive is $1 - 0.95^{20} = 64\%$.

~~~dg-fig 13.06-multiple.svg
Left: the chance of at least one false alarm climbs to 64% at 20 tests; Bonferroni holds it at 5%. Right: Benjamini-Hochberg compares the k-th smallest p-value to the rising line k·q/m and rejects everything up to the last point under the line.
~~~

Two standard fixes, for $m$ hypotheses:

- **Bonferroni** controls the **family-wise error rate (FWER)**, the probability of any false positive, by testing each hypothesis at $\alpha/m$. Simple and valid under any dependence, but conservative for large $m$.
- **Benjamini-Hochberg (BH)** controls the **false discovery rate (FDR)**, the expected fraction of rejections that are false. Sort the p-values $p_{(1)} \le \dots \le p_{(m)}$, find the largest $k$ with $p_{(k)} \le \frac{k}{m}q$, and reject hypotheses $1..k$. Here $q$ is the FDR level you accept, often 0.05 or 0.10.

<div class="ex">
<div class="lab">Worked example</div>

**five secondary metrics, by hand**

Sorted p-values: 0.001, 0.012, 0.021, 0.040, 0.300, with $m = 5$ and $q = 0.05$.

BH thresholds $k q/m$: $1 \times 0.01 = 0.010$, then 0.020, 0.030, 0.040, 0.050.

Compare from the top: $0.300 > 0.050$, no; $0.040 \le 0.040$, yes. So the largest $k$ is 4, and BH rejects the first four (0.012 is not below 0.010 on its own, but step-up rejects everything below the largest passing rank).

Bonferroni tests each at $0.05/5 = 0.01$ and rejects only the first.

</div>

| Sorted p-value | 0.001 | 0.012 | 0.021 | 0.040 | 0.300
| --- | --- | --- | --- | --- | --- |
| BH threshold $kq/m$ ($q = 0.05$, $m = 5$) | 0.010 | 0.020 | 0.030 | 0.040 | 0.050
| BH rejects | yes | yes | yes | yes | no
| Bonferroni ($p \le 0.01$) rejects | yes | no | no | no | no

Use FWER control for the primary decision and FDR for screening secondary metrics or segments; a segment found after looking is a hypothesis for the next test.

## Step 6: shrink the noise with CUPED

The sample-size formula has $\sigma^2$ on top. Lower the variance and you need fewer users. **CUPED** (Controlled-experiment Using Pre-Experiment Data) does this with data you already have: each user's value of the same metric in the weeks before the test.

Heavy spenders last month tend to be heavy spenders this month, whichever arm they are in; that predictable part is noise for the comparison, so subtract it. Analyse the adjusted metric

$$Y_{\text{cuped}} = Y - \theta\,(X - \bar X), \qquad \theta = \frac{\operatorname{Cov}(X, Y)}{\operatorname{Var}(X)},$$

where $Y$ is the metric during the test, $X$ is the pre-period covariate, $\bar X$ its overall mean and $\theta$ the regression slope of $Y$ on $X$. Because $X$ was measured before assignment, it cannot be affected by the treatment, so the mean difference between arms is unchanged. The variance is multiplied by $1 - \rho^2$, where $\rho$ is the correlation between $X$ and $Y$.

~~~dg-fig 13.06-cuped.svg
Left: spend during the test against spend the month before (correlation 0.70). Right: 3,000 simulated experiments with a true effect of +0.3; CUPED estimates cluster around the truth with variance 0.51 times that of the plain difference.
~~~

With $\rho = 0.7$, $1 - 0.49 = 0.51$: the variance roughly halves, and so does the sample size needed for the same power. New users have no pre-period data; give them $X = \bar X$ (no adjustment).

## The whole test in code

```python
import numpy as np
from scipy import stats

# 1. Design: users per arm for 5.0% -> 5.5% at alpha = 0.05 (two-sided), power 0.80
z_a, z_b = stats.norm.ppf(0.975), stats.norm.ppf(0.80)
p1, p2 = 0.05, 0.055
n = (z_a + z_b) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2
print(round(n))                                   # 31231

# 2. Sanity check: sample-ratio mismatch on the counts actually logged
counts = [31250, 31250]
print(stats.chisquare(counts).pvalue)             # 1.0 (no SRM)

# 3. Analysis: two-proportion z-test and 95% CI for the difference
xA, nA, xB, nB = 1571, 31250, 1734, 31250
pA, pB = xA / nA, xB / nB
pool = (xA + xB) / (nA + nB)
z = (pB - pA) / np.sqrt(pool * (1 - pool) * (1 / nA + 1 / nB))
p_value = 2 * stats.norm.sf(abs(z))
se = np.sqrt(pA * (1 - pA) / nA + pB * (1 - pB) / nB)
ci = (pB - pA - z_a * se, pB - pA + z_a * se)
print(round(z, 2), round(p_value, 4))             # 2.91 0.0036
print(np.round(np.array(ci) * 100, 2))            # [0.17 0.87] percentage points

# 4. Secondary metrics: Benjamini-Hochberg adjusted p-values
p_secondary = [0.001, 0.012, 0.021, 0.040, 0.300]
print(stats.false_discovery_control(p_secondary)) # [0.005 0.03 0.035 0.05 0.3]
```

`false_discovery_control` (SciPy 1.11+) returns BH-adjusted p-values: reject where the adjusted value is at most $q$, here the first four.

## Reading the result

Read the interval against the MDE, not just $p$ against 0.05:

- **Interval entirely above zero and near or above the MDE**: a clear win. Ship, after the guardrails pass.
- **Interval above zero but mostly below the MDE**: real but small; shipping is a cost question.
- **Interval straddling zero and narrow** (say $-0.1$ to $+0.1$ points): a confident "no meaningful effect", which is a useful result.
- **Interval straddling zero and wide**: the test could not tell. Do not report "no effect"; report "underpowered".

Before any of that: no SRM, guardrails flat or better, and A/A runs of the pipeline significant about 5% of the time.

## What goes wrong

Underpowered tests also exaggerate the wins they report: only lucky high draws clear the bar. This is the **winner's curse**.

~~~dg-fig 13.06-winners.svg
A true lift of 0.25 points tested with 31,250 users per arm (power about 29%) and with 122,000 per arm (power about 80%). Among significant results the small test reports +0.46 on average, almost double the truth; the well-powered test reports +0.28.
~~~

| Mistake | Symptom | Fix
| --- | --- | --- |
| Peeking and stopping at the first $p < 0.05$ | "Wins" that vanish on rerun; far more than 5% of A/A tests significant | Fixed horizon, or a planned sequential design
| Analysing page views of a user-randomised test | Intervals too narrow, too many significant results | Aggregate to the user, or use clustered or delta-method SEs
| Twenty metrics and a dozen segments, no correction | A surprising segment "win" every week | One primary metric; BH for screening; retest segments
| Underpowered test | Mostly flat results; the significant ones overstate the lift | Power calculation; CUPED; a larger MDE or longer run
| Ignoring SRM | Effects caused by who got logged, not by the change | Chi-square SRM check before reading any metric
| Users interact across arms (marketplace, social) | Treatment changes control users' outcomes | Randomise clusters (cities, groups) or switchback designs

## Going deeper

**Where the sample-size formula comes from.** Under the null the estimate $\hat\delta$ is approximately $N(0, \text{SE}^2)$, and we reject when $\hat\delta > z_{1-\alpha/2}\text{SE}$ (ignoring the negligible far tail). Under the alternative $\hat\delta \sim N(\delta, \text{SE}^2)$, so power is $\Phi(\delta/\text{SE} - z_{1-\alpha/2})$. Setting this to $1-\beta$ gives $\delta/\text{SE} = z_{1-\alpha/2} + z_{1-\beta}$; substitute $\text{SE}^2 = (\sigma_A^2+\sigma_B^2)/n$ and solve. Unequal splits cost power: with fractions $f$ and $1-f$ of $N$ users, the variance is $\sigma^2/(N f(1-f))$, which is smallest at $f = 0.5$.

**Alpha spending.** A group-sequential design with $K$ looks chooses bars $c_1, \dots, c_K$ so the chance of crossing any under the null is $\alpha$. For five looks at $\alpha = 0.05$, Pocock uses a constant 2.41; O'Brien-Fleming uses $c\sqrt{K/k}$ with $c \approx 2.04$: 4.56, 3.23, 2.63, 2.28, 2.04 (standard values, matched by simulation). Always-valid inference (mSPRT) instead builds a likelihood ratio that is a martingale under the null, and Ville's inequality bounds the chance it ever exceeds $1/\alpha$.

**Why BH works.** Under independent (or positively dependent) test statistics, BH controls FDR at $q \cdot m_0/m \le q$, where $m_0$ is the number of true nulls. Under arbitrary dependence, the Benjamini-Yekutieli variant divides $q$ by $\sum_{i=1}^m 1/i$ (`method='by'` in SciPy). Holm's step-down procedure controls FWER and is uniformly more powerful than Bonferroni.

**CUPED is regression adjustment.** Choosing $\theta$ to minimise $\operatorname{Var}(Y - \theta X)$ gives $\theta = \operatorname{Cov}(X,Y)/\operatorname{Var}(X)$ and the minimum $\operatorname{Var}(Y)(1-\rho^2)$. Using the pooled $\theta$ from both arms is equivalent to ANCOVA with a treatment indicator, and it remains unbiased in large samples because $X$ is independent of assignment. Several covariates generalise it to multiple regression.

**Ratio metrics.** For clicks per page view under user randomisation, numerator and denominator are per-user sums; the delta method (or a user-level bootstrap) gives the right variance.

## Key terms

| Term | Meaning
| --- | --- |
| A/B test | A randomised experiment comparing a control arm with a treatment arm on a metric
| Randomisation unit | The thing assigned to an arm (usually the user); analyse at this unit
| Minimum detectable effect (MDE) | The smallest true effect the test is sized to detect with the planned power
| Power | Chance the test is significant when the true effect equals the MDE; usually 80%
| Sample-ratio mismatch (SRM) | An arm-size imbalance too large for chance, signalling broken assignment or logging
| Peeking | Repeated looks with the option to stop at the first significant result, which inflates false positives
| FWER / FDR | Chance of any false positive / expected fraction of false positives among rejections
| Benjamini-Hochberg | Step-up procedure comparing sorted p-values with $kq/m$; controls FDR
| CUPED | Subtracting a pre-period prediction to cut variance by a factor of $1-\rho^2$
| Winner's curse | Significant results from underpowered tests overstate the true effect

<div class="take">
<div class="lab">Key takeaway</div>

an A/B test is decided at design time: fix the unit, one primary metric, the MDE and the sample size from a power calculation, then analyse once (or with a proper sequential method), check SRM first, read the confidence interval against the MDE, and correct for every extra metric or segment you look at.

</div>
