---
id: "09.03"
course: "machine-learning"
module: "09"
slug: "monitoring-models"
title: "Monitoring models"
keyIdeas: ["Labels in production often arrive days or months late, so monitoring layers fast proxies (data quality, input drift, prediction drift) on top of slow truth (accuracy and calibration once labels mature).", "**PSI** compares binned distributions: $\\text{PSI} = \\sum_b (a_b - e_b)\\ln\\frac{a_b}{e_b}$, with bins fixed from reference quantiles. The usual reading is below 0.1 stable, 0.1 to 0.25 worth a look, above 0.25 a real shift.", "The **KS** statistic $D = \\sup_x |F_{\\text{ref}}(x) - F_{\\text{cur}}(x)|$ is distribution-free, but with large samples its p-value flags trivial shifts; alert on the effect size $D$, not on $p < 0.05$.", "Compute label-based metrics only on **matured cohorts**: recent cohorts look artificially good because their labels have not arrived yet.", "**Calibration drift** (ECE rising while AUC holds) breaks every downstream threshold and expected-value calculation even when ranking is fine; label shift causes it readily.", "Alerts should require persistence (several consecutive breaching windows), use thresholds set from known-good history, and route to an owner with a runbook; a drift alert with no owner is noise."]
question: "A churn model's AUC on last quarter's matured labels is unchanged at 0.79, but the retention team reports that the 'top 10% risk' list now converts to actual churn at 18% instead of the 30% it used to. Explain how both facts can be true and what you would monitor to have caught it earlier."
answer: "AUC measures ranking only, and is unchanged if the model still orders customers equally well. The retention list depends on calibration and the base rate. If overall churn fell (label shift, perhaps from a pricing change or a better onboarding flow), the top decile contains fewer real churners even with exactly the same ordering, and the model's probabilities are now over-confident: rescaling everyone's odds by the same factor changes precision in the top decile but never changes AUC. Alternatively, covariate shift may have moved the score distribution so that the 'top 10%' now reaches further into lower-risk customers. Either way the probability no longer means what it meant when the list was designed. Monitoring that would have caught it earlier: (1) the churn base rate on matured cohorts, plus an early proxy such as 30-day cancellations compared with what older cohorts showed at the same age; (2) ECE or a reliability curve per monthly cohort, which would show the curve dropping below the diagonal; (3) PSI of the prediction score and the share of customers above the action threshold, which move within days rather than a quarter; (4) precision in the top decile as a business metric with a persistent-breach alert (for example three consecutive weeks below 25%) and a named owner. The usual fix is to recalibrate on recent matured data (for example Platt or isotonic scaling, or a prior-shift odds correction), not a full retrain, and to check upstream data quality before doing either."
sources: [{"title": "Rabanser, Günnemann and Lipton, Failing Loudly: Detecting Dataset Shift (2018)", "url": "https://arxiv.org/abs/1810.11953", "note": "Empirical comparison of shift detectors, including univariate KS with Bonferroni and classifier-based tests."}, {"title": "Chip Huyen, Data Distribution Shifts and Monitoring", "url": "https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html", "note": "Covariate, label and concept shift with production examples and what to monitor."}, {"title": "SciPy ks_2samp reference", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html", "note": "The statistic, alternatives and exact versus asymptotic p-values."}, {"title": "Guo et al., On Calibration of Modern Neural Networks (2017)", "url": "https://arxiv.org/abs/1706.04599", "note": "Defines ECE and reliability diagrams as used for calibration monitoring."}]
practice: ["population-stability-index", "two-sample-ks-drift", "expected-calibration-error", "delayed-label-monitoring", "consecutive-breach-alerts"]
tags: ["drift", "calibration", "monitoring"]
tools: ["NumPy", "SciPy"]
---

## What you will learn

- Why a deployed model fails silently, and which signals you can watch at which delay: data quality, input drift, prediction drift, and finally labels.
- How to measure drift with the **Population Stability Index (PSI)** and the **Kolmogorov-Smirnov (KS)** statistic, by hand and in code, and how to read their values.
- Why a p-value is the wrong thing to alert on when you have millions of rows.
- How delayed labels distort performance numbers, and how matured cohorts fix that.
- Why calibration can drift while AUC stays flat, and why that still breaks decisions.
- How to set alert thresholds and persistence rules that people trust instead of mute.

*Before you start:* you should know probabilities as model outputs and reliability diagrams (03.04), AUC and thresholds (06.01, 06.02), and why training and serving must see the same features (09.01). Logarithms and cumulative distributions are the only maths needed.

Lesson 09.02 made training data reproducible, so you can rebuild the model you shipped. This lesson is about what happens after it ships. A model does not crash when the world changes; it keeps returning numbers that quietly become wrong. Monitoring is how you notice first.

## The idea in one picture

A lender's approval model was trained on last year's applications. One input is `loan_amount`. During training, the team split the amounts into five bins so that each bin held exactly 20% of applications (the **quintiles** of the training data). This week, after a marketing campaign aimed at larger loans, the same bins hold different shares:

| Bin (loan amount) | under 4k | 4k-7k | 7k-11k | 11k-17k | over 17k
| --- | --- | --- | --- | --- | --- |
| Training share | 20% | 20% | 20% | 20% | 20%
| This week's share | 10% | 14% | 18% | 24% | 34%

Nothing errored, but the population has moved towards big loans, where the model learned from fewer examples.

~~~dg-fig 09.03-psi-bins.svg
Left: training and this week's shares in the same five bins. Right: each bin's PSI term. The two outer bins, where the shares moved most in relative terms, contribute almost all of the 0.174 total.
~~~

One number summarises this picture: the PSI, here **0.174**. By the common rule of thumb that is "worth a look" (0.1 to 0.25), not yet "a real shift". Most of it comes from the smallest-loan bin halving and the largest growing by 70%.

The **reference** distribution is the known-good data the model was trained or validated on; the **current** distribution is a recent window of traffic (an hour, a day, a week). **Drift** is any change between them: a fact about data, separate from whether it hurts the model.

## Step 1: know what you can see, and when

A model can go wrong in ways that raise no exception: an upstream column changes meaning, the customer mix shifts, or the link between inputs and outcome changes. What you can observe depends on how long you wait.

~~~dg-flow Signals become available at very different delays; monitoring needs a layer for each.
Request -(seconds)-> Data quality checks -(minutes)-> Input and prediction drift -(days to months)-> *Label-based metrics
~~~

- **Data quality** (immediately): null rates, schema violations, out-of-range values, category counts. Cheapest and highest-yield: most incidents are broken pipelines, not subtle drift.
- **Input drift** (minutes to hours): has the distribution of each feature moved away from the reference?
- **Prediction drift** (minutes to hours): has the distribution of scores, or the share of positive decisions, moved? This is often the single best early proxy, because the score summarises every feature through the model, weighted by what the model actually cares about.
- **Performance** (when labels arrive): accuracy, AUC, calibration, computed on predictions whose true outcomes are now known.

It helps to name the kinds of shift. Write $x$ for the inputs and $y$ for the outcome.

- **Covariate shift**: $P(x)$ changes but $P(y \mid x)$ stays fixed. Our loan campaign is an example: more big loans, but a given applicant is as risky as before.
- **Label shift**: $P(y)$ changes, for example the fraud base rate doubles, while each class still looks the same.
- **Concept drift**: $P(y \mid x)$ itself changes. The same applicant is now riskier (a recession), or fraudsters change tactics. Only this kind makes the model wrong on inputs it handles well today, and it is visible only through labels.

## Step 2: measure drift on one feature with PSI

In words: split the feature into bins, compare the share of reference and current data in each bin, and add up a penalty that grows with both the absolute and the relative change.

$$\text{PSI} = \sum_{b=1}^{B} (a_b - e_b)\,\ln\frac{a_b}{e_b}$$

- $B$ is the number of bins (10 is the usual choice; 5 above to keep the table small).
- $e_b$ is the share of **reference** data in bin $b$ ("expected").
- $a_b$ is the share of **current** data in bin $b$ ("actual").
- $\ln$ is the natural logarithm.

Every term is zero or positive: if $a_b > e_b$ both factors are positive, and if $a_b < e_b$ both are negative. So PSI is never negative, and it is 0 only when every bin matches.

Build the bins from **quantiles of the reference**, so each bin starts with the same share. If a bin is empty in either sample, $\ln(a_b/e_b)$ is infinite, so add a tiny epsilon to every share. For a categorical feature, use one bin per category plus an "other" bin for categories never seen in training.

The usual reading, which is a convention from credit scoring rather than a law:

| PSI | Usual reading | Typical action
| --- | --- | --- |
| below 0.1 | stable | nothing
| 0.1 to 0.25 | moderate shift | look at it, check upstream and segments
| above 0.25 | major shift | investigate now; consider retraining or recalibrating

<div class="ex">
<div class="lab">Worked example</div>

**PSI by hand on four bins**

The reference splits into quartiles, so $e = (0.25, 0.25, 0.25, 0.25)$. This week's shares are $a = (0.15, 0.20, 0.30, 0.35)$.

Bin 1: $(0.15 - 0.25)\ln(0.15/0.25) = (-0.10)\ln(0.6) = (-0.10)(-0.511) = 0.0511$.

Bin 2: $(0.20 - 0.25)\ln(0.8) = (-0.05)(-0.223) = 0.0112$.

Bin 3: $(0.30 - 0.25)\ln(1.2) = (0.05)(0.182) = 0.0091$.

Bin 4: $(0.35 - 0.25)\ln(1.4) = (0.10)(0.336) = 0.0336$.

Total: $0.0511 + 0.0112 + 0.0091 + 0.0336 = 0.105$. Just over 0.1: a moderate shift worth a look.

Notice bins 1 and 4. Both moved by 0.10 in absolute terms, but bin 1 contributes more (0.051 versus 0.034) because losing 40% of its mass is a bigger relative change than gaining 40%. The log ratio is what makes PSI care about relative change.

</div>

## Step 3: measure drift without bins, with KS

PSI depends on where you put the bin edges and cannot see changes inside a bin. For a continuous feature, the two-sample **Kolmogorov-Smirnov** test avoids binning. It compares the two **empirical CDFs**: for each value $x$, $F(x)$ is the share of the sample at or below $x$.

$$D = \sup_x \,\big|F_{\text{ref}}(x) - F_{\text{cur}}(x)\big|$$

- $F_{\text{ref}}$ and $F_{\text{cur}}$ are the empirical CDFs of the reference and current samples.
- $\sup_x$ means "the largest value over all $x$".
- $D$ is between 0 (identical) and 1 (no overlap at all). It needs no distribution assumptions.

Small check: reference $\{1, 2, 3, 4\}$, current $\{3, 4, 5, 6\}$. At $x = 2$, half the reference is at or below 2 and none of the current sample is, so the gap is $0.5 - 0 = 0.5$. No point does better, so $D = 0.5$.

~~~dg-fig 09.03-ks.svg
Left: D is the largest vertical gap between the two CDFs. Right: a fixed, harmless shift of 0.02 standard deviations. As the samples grow, D settles near 0.008, but the p-value collapses below 0.05 from about 10,000 rows and reaches 10⁻²⁸ at a million.
~~~

The right panel is the trap. The KS test also returns a p-value, which answers "could these two samples come from exactly the same distribution?" With a million rows, almost nothing in production is *exactly* the same, so the answer is always no. The p-value measures how much data you have as much as how big the shift is. The original lesson's rule stands: with a million rows on each side, a $D$ of 0.003 can be "significant" at $p < 0.001$ and still mean nothing. Alert on the effect size $D$ (for example $D > 0.1$) and use the p-value only on small samples.

## Step 4: watch the predictions, not only the features

200 features give 200 drift numbers every hour, some always red by chance. Two habits keep this manageable.

First, track drift on the **model's output**: score PSI, the share above the decision threshold, the mean predicted probability. These weight every input by its effect on the decision.

Second, rank feature drift by feature importance. Drift on an unimportant feature is harmless; drift on the top feature is serious. Use per-feature drift to *explain* a prediction-drift alert, not as a wall of separate alerts.

<div class="ex">
<div class="lab">Worked example</div>

**reading a drift dashboard**

A loan model's top feature, `debt_to_income`, has PSI 0.04 this week. The prediction score distribution has PSI 0.31, and the share of applications above the approval threshold rose from 38% to 52%.

So the score moved a lot while the most important input did not. Sorting feature PSI from largest to smallest finds the cause: a mid-importance feature, `employment_length`, has PSI 0.6. An upstream change started sending years instead of months, so every applicant suddenly looked like a short-tenure worker in the training scale.

Prediction drift found it; per-feature PSI explained it; a schema range check (09.01) would have prevented it.

</div>

~~~dg-fig 09.03-psi-weekly.svg
Weekly PSI against training data for the score and one feature. The holiday spike in week 9 is a single breach and does not alert. From week 19 the score stays above 0.25, and the alert fires at the third consecutive breach, in week 21, with the drifting feature already pointing at the cause.
~~~

## Step 5: performance needs labels, and labels arrive late

Drift tells you the inputs changed. Only labels tell you whether the model got worse. In many products the label is the slow part: card chargebacks arrive up to 60 to 120 days after a transaction, and loan defaults take months.

~~~dg-fig 09.03-maturation.svg
Left: the share of chargebacks that have arrived, by days since the transaction (about 70% at 30 days, 97% at 90). Right: measured today, recent weekly cohorts look almost fraud-free only because their labels have not arrived; a real rise in fraud in the last six weeks is hidden under the immaturity.
~~~

If you compute "fraud rate this month" naively, recent weeks always look better than they are, and the effect is large: a cohort observed at 30 days shows only about 70% of its eventual fraud, so a true rate of 0.8% looks like 0.56%. Handle the lag explicitly:

1. Compute label-based metrics only on **matured cohorts**: predictions old enough that nearly all their labels have arrived (here, older than 90 days).
2. Track **early proxies** with a known lag: first-week disputes as a leading indicator of 90-day fraud.
3. Publish the **maturation curve** so that a dip caused by immature labels is not mistaken for improvement, and a rise is not hidden.

## Step 6: calibration can drift while AUC holds

If a model outputs 0.2, about 20% of such cases should be positive. **Expected calibration error** summarises how far that holds. Bin predictions by their probability, and in each bin compare the average prediction ("confidence") with the observed positive rate:

$$\text{ECE} = \sum_{b} \frac{n_b}{N}\,\big|\text{acc}_b - \text{conf}_b\big|$$

- $n_b$ is the number of predictions in bin $b$, and $N$ the total.
- $\text{conf}_b$ is the mean predicted probability in bin $b$.
- $\text{acc}_b$ is the observed rate of positives in bin $b$.

Number check with three bins: 600 cases predicted at 0.05 with 8% positive, 300 at 0.30 with 36% positive, 100 at 0.70 with 78% positive. ECE $= 0.6 \times 0.03 + 0.3 \times 0.06 + 0.1 \times 0.08 = 0.018 + 0.018 + 0.008 = 0.044$.

Calibration matters whenever someone uses the probability itself: expected loss, a bid price, a fixed approval threshold (03.04 covers reliability diagrams and recalibration). It drifts readily under label shift. If the fraud base rate doubles, a model that still ranks well becomes under-confident everywhere: AUC barely moves, and a fixed threshold silently lets more fraud through.

~~~dg-fig 09.03-calibration.svg
Left: the fraud base rate climbs from 5% to 12% over a year. AUC on matured cohorts stays flat near 0.90, while ECE crosses its 0.02 alert line around month 6. Right: in month 12 the reliability curve sits above the diagonal, so every predicted probability is too low.
~~~

AUC only looks at the order of scores, and under pure label shift the order is unchanged, so an AUC-only dashboard shows nothing here.

## Monitoring in code

A small toolkit: PSI, KS, ECE and a persistence rule. The printed numbers come from running this exact code.

```python
import numpy as np
from scipy.stats import ks_2samp

def psi(ref, cur, bins=10, eps=1e-6):
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf          # catch values outside the training range
    e = np.histogram(ref, edges)[0] / len(ref) + eps
    a = np.histogram(cur, edges)[0] / len(cur) + eps
    return float(np.sum((a - e) * np.log(a / e)))

def ece(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
               for b in range(bins) if (idx == b).any())

def persistent(breach, k=3):                       # True once k windows in a row breach
    run, out = 0, []
    for b in breach:
        run = run + 1 if b else 0
        out.append(run >= k)
    return out

rng = np.random.default_rng(0)
ref   = rng.lognormal(9.0, 0.6, 50_000)            # loan amounts at training time
same  = rng.lognormal(9.0, 0.6, 5_000)             # a healthy week
moved = rng.lognormal(9.2, 0.6, 5_000)             # a week with bigger loans
print(round(psi(ref, same), 4), round(psi(ref, moved), 4))      # 0.0021 0.1136
r1, r2 = ks_2samp(ref, same), ks_2samp(ref, moved)
print(round(r1.statistic, 3), round(r2.statistic, 3))           # 0.015 0.141
print(f"{r2.pvalue:.1e}")                                       # 8.4e-79

p = rng.uniform(0, 1, 20_000)
y_ok  = (rng.uniform(0, 1, 20_000) < p).astype(int)            # calibrated outcomes
y_new = (rng.uniform(0, 1, 20_000) < np.minimum(1, 1.3 * p)).astype(int)
print(round(ece(p, y_ok), 3), round(ece(p, y_new), 3))          # 0.007 0.115

weekly_psi = [0.03, 0.29, 0.04, 0.27, 0.31, 0.33, 0.36]
print(persistent([v > 0.25 for v in weekly_psi]))
# [False, False, False, False, False, True, True]
```

The KS p-value of $10^{-79}$ would be just as overwhelming for a far smaller shift; $D = 0.141$ and PSI $= 0.114$ are the numbers to reason with. In production these run on a schedule against a fixed reference snapshot.

## Reading the result: alerts people trust

An alert that fires every day gets muted, and then it is worse than nothing. Four rules:

- **Require persistence.** Alert on, say, three consecutive windows above threshold, not on a single spike. In the weekly figure this ignored a holiday blip and still caught the real shift two weeks after it began.
- **Set thresholds from your own history.** Compute PSI between consecutive weeks of known-good data to see the natural noise floor. With 10 bins and 1,000 rows per side, two samples from the *same* distribution already average a PSI of about 0.018, and the 99th percentile is about 0.04. Small windows need higher thresholds.
- **Give every alert an owner and a runbook**: check upstream pipelines first, compare with the previous model, segment by channel or product, then choose rollback, recalibration or retraining.
- **Do not retrain by reflex.** Retraining on data from a broken pipeline bakes the bug into the next model. Recalibrating on recent matured data is often enough when only the base rate moved.

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Alerting on KS p-values | dozens of "significant" features every day on large traffic | alert on $D$ or PSI with thresholds from history
| Bins from current data, or equal-width bins on skewed data | PSI jumps when nothing changed; one bin holds 90% of rows | fix bin edges once from reference quantiles, store them with the model
| Metrics on immature labels | recent performance looks great, then gets revised down | report on matured cohorts plus early proxies
| Watching AUC only | ranking fine, decisions wrong, business metric falls | add ECE, base rate and top-decile precision
| Feedback loop ignored | labels exist only for approved cases | keep a small random holdout or exploration slice to see outcomes of rejected cases
| Single-window alerts | pages at 3 a.m. for noise | require consecutive breaches

## Going deeper

**What PSI is mathematically.** Write it as $\sum_b a_b \ln(a_b/e_b) + \sum_b e_b \ln(e_b/a_b)$. That is $\text{KL}(a \,\|\, e) + \text{KL}(e \,\|\, a)$, the symmetrised Kullback-Leibler divergence (the Jeffreys divergence) between the binned distributions. For small changes, $(a_b - e_b)\ln(a_b/e_b) \approx (a_b - e_b)^2 / e_b$, so PSI behaves like a chi-squared statistic per row. That gives the noise floor: when both samples come from the same distribution, $\mathbb{E}[\text{PSI}] \approx (B - 1)\left(\frac{1}{n_{\text{ref}}} + \frac{1}{n_{\text{cur}}}\right)$. With $B = 10$ and 1,000 rows each side that is $9 \times 0.002 = 0.018$, matching the simulation above. Fixed thresholds like 0.1 are only meaningful once windows are large enough that this floor is small.

**Persistence as a false-alarm budget.** If a healthy window breaches by chance with probability 0.05, and windows are roughly independent, three in a row happen with probability $0.05^3 = 1.25 \times 10^{-4}$. Persistence trades a delay of $k - 1$ windows for a much lower false-alarm rate. For faster detection of small, sustained changes, sequential methods such as **CUSUM** accumulate evidence: $S_t = \max(0, S_{t-1} + x_t - \mu_0 - \delta)$, alerting when $S_t > h$, where $\mu_0$ is the healthy mean, $\delta$ a tolerance and $h$ a threshold. Page-Hinkley is a close relative.

**Multivariate shift.** Each feature can look stable on its own while their combination changes (for example income and loan amount keep their marginals but stop being correlated). A **classifier two-sample test** catches this: label reference rows 0 and current rows 1, train a classifier, and check its held-out AUC. Near 0.5 means the samples are indistinguishable; well above 0.5 means shift, and the classifier's feature importances say where.

**Correcting for label shift.** If only the base rate moved from $\pi$ to $\pi'$ and class-conditional distributions are unchanged, the correct new probability follows from Bayes' rule by rescaling the odds: $\text{odds}' = \text{odds} \times \frac{\pi'/(1-\pi')}{\pi/(1-\pi)}$. A score of 0.2 trained at a 5% base rate becomes about 0.39 at 12%. This is why the calibration figure's AUC stayed flat: rescaling odds by a constant never changes the order.

## Key terms

| Term | Meaning
| --- | --- |
| Reference / current window | The known-good data you compare against, and the recent production data you check
| Covariate, label, concept shift | $P(x)$ changes; $P(y)$ changes; $P(y \mid x)$ changes
| PSI | Sum over bins of $(a_b - e_b)\ln(a_b/e_b)$; below 0.1 stable, above 0.25 major shift
| KS statistic $D$ | Largest gap between two empirical CDFs; an effect size, unlike its p-value
| Prediction drift | Change in the distribution of model scores or decisions; the best early proxy
| Matured cohort | Predictions old enough that nearly all of their labels have arrived
| ECE | Weighted average gap between predicted probability and observed rate across bins
| Persistence rule | Alert only after several consecutive breaching windows

<div class="take">
<div class="lab">Key takeaway</div>

monitor in layers by delay: data quality and prediction drift for speed, per-feature PSI and KS effect sizes for diagnosis, and calibration and accuracy on matured labels for truth, with thresholds set from your own history and persistent, owned alerts.

</div>
