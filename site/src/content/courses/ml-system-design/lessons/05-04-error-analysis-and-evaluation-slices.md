---
id: "05.04"
course: "ml-system-design"
module: "05"
slug: "error-analysis-and-evaluation-slices"
title: "Error analysis and evaluation slices"
keyIdeas: ["Aggregate metrics hide where a model fails. **Slices**, such as segments by country, device, user tenure or item type, show which groups are underserved.", "**Error analysis** means reading misclassified examples, grouping them into causes and counting each cause, so you fix the biggest problem first.", "Common causes are label noise, missing features, distribution gaps, and bugs in data or serving. Each needs a different fix: more data, new features, relabelling or engineering.", "Turn recurring failure types into permanent test slices that every new model must pass."]
question: "A speech-to-text model for a customer-service product has 8% word error rate overall, but complaints are rising. Describe how you would find where it fails and decide what to fix first."
answer: "Slice the error rate by every attribute available: language and accent or region, audio channel (phone vs app), background noise level estimated from the audio, call length, speaker role (agent vs customer), and domain vocabulary such as product names and account numbers. Look for slices with much higher error and enough volume to matter, computing confidence intervals so small slices are not over-interpreted. Then sample 200 errors from the worst high-volume slices and label each with a cause: mis-heard product names, numbers formatted wrongly, crosstalk, low-bandwidth audio, labelling errors in references. Count causes and estimate how much each would reduce errors if fixed. For example, if 35% of customer-side errors are product names and account numbers, a custom vocabulary or contextual biasing gives the biggest win; if phone audio has twice the error, add phone-quality training data or augmentation. Add each important slice to the evaluation gates."
sources: [{"title": "Andrew Ng, Machine Learning Yearning (error analysis chapters)", "url": "https://info.deeplearning.ai/machine-learning-yearning-book", "note": "Manual error analysis and prioritisation."}, {"title": "Chung et al., Automated Data Slicing for Model Validation (Slice Finder, 2019)", "url": "https://arxiv.org/abs/1807.06068", "note": "Finding underperforming slices automatically."}, {"title": "Ribeiro et al., Beyond Accuracy: Behavioral Testing of NLP Models with CheckList (2020)", "url": "https://arxiv.org/abs/2005.04118", "note": "Turning failure types into tests."}]
practice: ["permutation-feature-importance", "group-by", "bootstrap-confidence-interval", "expected-calibration-error"]
tags: ["error analysis", "slices"]
tools: ["pandas"]
---

## Averages hide failures

A model with 95% accuracy can be 99% accurate for most users and 70% accurate for one country, device or language. Those users are the ones who complain, churn or are treated unfairly. Aggregate metrics are the starting point of evaluation, not the end.

## Slicing

Compute every key metric by segment:

- **user**: new versus returning, country, language, accessibility settings;
- **item or content**: category, popularity tier, age, creator size;
- **context**: platform, app version, time of day, network quality;
- **input properties**: length, noise, rare words, missing features.

Look for slices that are both worse than average and large enough to matter. Compute confidence intervals; a slice with 30 examples can look terrible by chance.

```python
import pandas as pd
def slice_report(df, by, metric="correct", min_n=200):
    g = df.groupby(by)[metric].agg(["mean", "count"])
    g["se"] = (g["mean"] * (1 - g["mean"]) / g["count"]) ** 0.5
    overall = df[metric].mean()
    g["gap"] = g["mean"] - overall
    return g[g["count"] >= min_n].sort_values("gap")
```

## Reading errors

Numbers tell you where; examples tell you why. Sample 100 to 200 errors from the most important slice, read them, and tag each with a cause. Then count.

| Cause | Share of errors | Fix
| --- | --- | --- |
| Label is wrong | 18% | relabel; improve guidelines
| Missing information in features | 30% | new feature or data source
| Rare pattern, few training examples | 25% | targeted data collection, augmentation
| Bug in preprocessing or serving | 12% | engineering fix
| Genuinely ambiguous | 15% | accept, or change the product

The table turns vague worries into a prioritised plan. The biggest bucket with a feasible fix goes first. A fix that addresses 30% of errors is worth more than a clever modelling idea that addresses 5%.

~~~dg-flow Error analysis turns metrics into a prioritised list of fixes and new tests.
Metrics by slice -(worst large slices)-> Sample errors -(tag causes)-> Count by cause -(largest fixable)-> *Fix -(add slice + cases)-> Evaluation gates
~~~

## Estimating the ceiling

For each cause, estimate the best possible gain: if every label error were fixed, how much would the metric improve? This "ceiling analysis" stops teams from spending months on a component that could only add half a point.

<div class="ex">
<div class="lab">Worked example</div>

**a slice that explained the complaints**

A food-delivery ETA model had good average error, but complaints came mainly from one city. Slicing by city and hour showed large under-estimates at dinner time in districts with new high-rise buildings. Reading cases revealed couriers spent 8 to 12 minutes getting from the street to the customer's door, which the model did not know about. A per-building "last-50-metres" feature learned from past deliveries halved the error in those districts.

</div>

## Automating slice discovery

With many attributes, manual slicing misses combinations ("Android, app version 5.2, in Brazil, at night"). Tools like Slice Finder search for subgroups with significantly worse performance. Treat their output as leads to investigate, not conclusions.

## Behavioural tests

Some failures are best expressed as tests rather than slices:

- **invariance**: changing a name or a neutral word should not change a sentiment label;
- **directional**: adding "not" should flip it;
- **minimum functionality**: simple cases every model must get right.

Add each discovered failure type as a test case or a slice in the evaluation gates (04.04), so it cannot come back unnoticed.

## Fairness slices

Slices by protected or sensitive attributes deserve particular care. Report error rates, false-positive rates and calibration per group where the data and policy allow it, and decide with policy and legal teams which differences are acceptable. When the attribute itself is not available, proxies such as region can reveal some disparities but should be interpreted cautiously.

## Sharing the results

Write error analysis up as a short table of causes, shares and planned fixes, with a few representative examples for each. It makes the next modelling decision easy to justify and lets people outside the team, such as product managers and reviewers, see why the plan targets one problem before another.

<div class="take">
<div class="lab">Key takeaway</div>

report every key metric by slice with confidence intervals, read and tag a sample of errors to find their causes, fix the largest fixable cause first, and turn each failure type into a permanent slice or test in the evaluation gates.

</div>
