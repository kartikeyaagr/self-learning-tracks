---
id: "09.03"
course: "machine-learning"
module: "09"
slug: "monitoring-models"
title: "Monitoring models"
keyIdeas: ["Labels in production often arrive days or months late, so monitoring layers fast proxies (input drift, prediction drift, data quality) on top of slow truth (accuracy and calibration once labels mature).", "**PSI** compares binned distributions: $\\text{PSI} = \\sum_b (a_b - e_b)\\ln\\frac{a_b}{e_b}$; the usual reading is below 0.1 stable, 0.1 to 0.25 worth a look, above 0.25 a real shift.", "The **KS** statistic $D = \\sup_x |F_{\\text{ref}}(x) - F_{\\text{cur}}(x)|$ is distribution-free, but with large samples its p-value flags trivial shifts; alert on the effect size $D$, not on $p < 0.05$.", "**Calibration drift** (ECE rising while AUC holds) breaks every downstream threshold and expected-value calculation even when ranking is fine.", "Alerts should require persistence (several consecutive breaching windows) and route to a person with a runbook; a drift alert with no owner is noise."]
question: "A churn model's AUC on last quarter's matured labels is unchanged at 0.79, but the retention team reports that the 'top 10% risk' list now converts to actual churn at 18% instead of the 30% it used to. Explain how both facts can be true and what you would monitor to have caught it earlier."
answer: "AUC measures ranking only, and is unchanged if the model still orders customers equally well. The retention list depends on calibration and the base rate: if overall churn fell (label shift, perhaps from a pricing change), the top decile contains fewer real churners even with the same ordering, so the model's probabilities are now over-confident. Alternatively, the score distribution may have shifted so that the top 10% now reaches further into lower-risk customers. Either way, the probability no longer means what it meant. Monitoring that would catch it: base-rate tracking on matured cohorts, ECE or a reliability curve per cohort, prediction-distribution PSI, and precision in the top decile as a business metric with a persistent-breach alert. The fix is usually recalibration on recent matured data, not a full retrain."
sources: [{"title": "Rabanser, Günnemann and Lipton, Failing Loudly: Detecting Dataset Shift (2018)", "url": "https://arxiv.org/abs/1810.11953", "note": "Empirical comparison of shift detectors, including univariate KS with Bonferroni and classifier-based tests."}, {"title": "Chip Huyen, Data Distribution Shifts and Monitoring", "url": "https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html", "note": "Covariate, label and concept shift with production examples and what to monitor."}, {"title": "SciPy ks_2samp reference", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html", "note": "The statistic, alternatives and exact versus asymptotic p-values."}, {"title": "Guo et al., On Calibration of Modern Neural Networks (2017)", "url": "https://arxiv.org/abs/1706.04599", "note": "Defines ECE and reliability diagrams as used for calibration monitoring."}]
practice: ["population-stability-index", "two-sample-ks-drift", "expected-calibration-error", "delayed-label-monitoring", "consecutive-breach-alerts"]
tags: ["drift", "calibration", "monitoring"]
tools: ["NumPy", "SciPy"]
---

## What can be watched, and when

A deployed model fails in ways that do not raise exceptions: an upstream column changes meaning, the population shifts, the relationship between features and outcome changes. What you can observe depends on time.

~~~dg-flow Signals become available at very different delays; monitoring needs a layer for each.
Request -(seconds)-> Data quality checks -(minutes)-> Input and prediction drift -(days to months)-> *Label-based metrics
~~~

- **Data quality** (immediately): null rates, schema violations, out-of-range values, category cardinality. The cheapest and highest-yield layer: most incidents are broken pipelines, not subtle drift.
- **Input drift** (minutes to hours): has the distribution of each feature moved away from training?
- **Prediction drift** (minutes to hours): has the distribution of scores, or the share of positive decisions, moved? This is often the single best proxy, because it summarises all features through the model.
- **Performance** (when labels arrive): accuracy, AUC, calibration on predictions whose labels have matured.

Types of shift, informally: *covariate shift* changes $P(x)$ with $P(y \mid x)$ fixed; *label shift* changes $P(y)$; *concept drift* changes $P(y \mid x)$ itself. Only the last one makes the model wrong for inputs it handles well today, and it is visible only through labels.

## Measuring drift: PSI and KS

**Population Stability Index**: bin a feature using quantiles of the reference (training) data, often 10 bins, compute the share of reference $e_b$ and current $a_b$ in each bin, then

$$\text{PSI} = \sum_{b} (a_b - e_b)\,\ln\frac{a_b}{e_b}$$

PSI is a symmetrised KL divergence on the binned distributions. Empty bins make it infinite, so add a small epsilon. It works on categorical features directly (one bin per category plus "other").

**Two-sample Kolmogorov-Smirnov**: $D = \sup_x |F_{\text{ref}}(x) - F_{\text{cur}}(x)|$, the largest gap between the two empirical CDFs. No binning and no distribution assumptions, for continuous features. The trap is sample size: with 1 million rows on each side, a $D$ of 0.003 is "significant" at $p < 0.001$ and means nothing. Alert on $D$ (for example, $D > 0.1$) and use the p-value only for small samples.

```python
import numpy as np
def psi(ref, cur, bins=10, eps=1e-6):
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(ref, edges)[0] / len(ref) + eps
    a = np.histogram(cur, edges)[0] / len(cur) + eps
    return float(np.sum((a - e) * np.log(a / e)))
```

| Method | Input | Strength | Weakness
| --- | --- | --- | --- |
| PSI | binned numeric or categorical | interpretable, one scale for all features | depends on binning; insensitive inside bins
| KS statistic | continuous | no binning, exact | p-value meaningless at scale; one feature at a time
| Page-Hinkley / CUSUM | a stream of a metric | detects the change point quickly | needs a tolerance and threshold tuned per stream
| Classifier two-sample test | all features jointly | catches multivariate shifts | heavier; needs care to interpret

Drift on an unimportant feature is harmless; drift on the top feature is serious. Weight drift alerts by feature importance, and always look at prediction drift too.

## Calibration drift and delayed labels

If a model outputs 0.2, about 20% of such cases should be positive. Expected calibration error summarises how far that holds: bin predictions by confidence, then $\text{ECE} = \sum_b \frac{n_b}{N} |\text{acc}_b - \text{conf}_b|$ (see ML lesson "Calibrated probabilities" for reliability diagrams and recalibration). Calibration matters whenever someone uses the probability itself: expected loss, bid prices, a fixed threshold. It drifts readily under label shift: if the fraud base rate doubles, a well-ranked model is now under-confident everywhere, AUC barely moves, and the approval threshold silently lets more fraud through.

Labels often lag. Chargebacks arrive up to 60 to 120 days after a transaction; loan defaults take months. Handle this explicitly:

1. Compute label-based metrics only on **matured cohorts**: predictions old enough that nearly all their labels have arrived.
2. Track early proxies with known lag: first-week disputes as a leading indicator of 90-day fraud.
3. Report the maturation curve so a dip caused by immature labels is not mistaken for degradation.

<div class="ex">
<div class="lab">Worked example</div>

**reading a drift dashboard**

A loan model's top feature, `debt_to_income`, has PSI 0.04 this week. The prediction score distribution has PSI 0.31, and the share of applications above the approval threshold rose from 38% to 52%. The input drift is in a mid-importance feature, `employment_length`, whose PSI is 0.6: an upstream change started sending years instead of months. Prediction drift found it; per-feature PSI explained it; a schema range check would have prevented it.

</div>

## Alerting that people trust

- Alert on **persistent** breaches, such as three consecutive hourly windows above threshold, not single spikes.
- Set thresholds from history: compute PSI between consecutive weeks of known-good data to see the natural noise floor.
- Each alert has an owner and a runbook: check upstream pipelines, compare against the previous model, retrain or roll back.
- Retraining is not automatically the fix. Retraining on data from a broken pipeline bakes the bug into the next model.

<div class="take">
<div class="lab">Key takeaway</div>

monitor in layers by delay: data quality and prediction drift for speed, per-feature PSI and KS effect sizes for diagnosis, and calibration and accuracy on matured labels for truth, with persistent, owned alerts.

</div>
