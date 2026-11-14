---
id: "15.06"
course: "ml-maths-course"
module: "15"
slug: "estimating-information-from-data"
title: "Estimating information from data"
keyIdeas: ["Plugging empirical frequencies into the entropy formula underestimates entropy with small samples; the bias is roughly $(m-1)/(2N)$ nats for $m$ observed values and $N$ samples.", "Mutual information estimates are biased upward for the same reason, so test them against a shuffled baseline.", "For continuous data, results depend on binning; nearest-neighbour estimators and neural bounds avoid bins but bring their own bias and variance.", "A model's predictive uncertainty splits into expected entropy (noise in the data) and mutual information between prediction and parameters (the model's own uncertainty)."]
question: "You run a model with dropout active 20 times on one input and average the predicted class distributions. (a) The averaged distribution has entropy 1.2 nats, and the average of the 20 individual entropies is 0.3 nats. What is the mutual information between the prediction and the model parameters, and what does it say? (b) For another input, both numbers are 1.2 nats. How do the two inputs differ? (c) Which input should you send for labelling first in active learning?"
answer: "(a) The mutual information is the total predictive entropy minus the expected entropy: $1.2 - 0.3 = 0.9$ nats. Each dropout sample is fairly confident (low individual entropy) but the samples disagree, so the uncertainty comes from the model, not the data. (b) For the second input the mutual information is 0: every dropout sample predicts the same spread-out distribution. The model agrees with itself that the input is ambiguous, which points to noise or genuine ambiguity in the data. (c) The first: labelling it can reduce the model's uncertainty. Labelling the second is unlikely to help because the ambiguity is in the data itself."
sources: [{"title": "Kraskov, Stoegbauer and Grassberger, Estimating Mutual Information (2003)", "url": "https://arxiv.org/abs/cond-mat/0305641", "note": "Nearest-neighbour estimation for continuous variables."}, {"title": "Poole et al., On Variational Bounds of Mutual Information (2019)", "url": "https://arxiv.org/abs/1905.06922", "note": "Bias and variance of neural mutual-information estimators."}, {"title": "Wikipedia, Entropy (information theory)", "url": "https://en.wikipedia.org/wiki/Entropy_(information_theory)", "note": "Includes estimation and related quantities."}, {"title": "MacKay, Information Theory, Inference, and Learning Algorithms (free book)", "url": "https://www.inference.org.uk/itprnn/book.pdf", "note": "Background for the Bayesian view of uncertainty."}]
practice: ["dropout-train-and-mc-predict", "bootstrap-confidence-interval", "expected-calibration-error", "k-fold-cross-validation"]
tags: ["entropy estimation", "bias", "predictive uncertainty", "mutual information estimators"]
tools: ["NumPy", "scikit-learn"]
---

## Why estimation is hard

The formulas in this module assume known distributions. In practice you have samples. Plugging empirical frequencies into the formulas is the obvious estimator, and for large samples over few values it works. With many possible values and limited data it fails in a predictable direction.

## Bias of the plug-in entropy estimator

Values that never appear in the sample get probability zero, and values that appear by chance are over-represented, so the empirical distribution looks more concentrated than the truth. The plug-in entropy is therefore biased downward. The Miller-Madow correction adds approximately $(m - 1)/(2N)$ nats, where $m$ is the number of distinct observed values and $N$ the sample size. When $m$ is comparable to $N$, even the corrected estimate is poor, and you should report a range rather than a number.

For mutual information, $\hat{I} = \hat{H}(X) + \hat{H}(Y) - \hat{H}(X, Y)$, and the joint table has many more cells than either margin, so its entropy is underestimated most. The result is positive mutual information between independent variables.

<div class="ex">
<div class="lab">Worked example</div>

**information that was not there**

A team ranked 3,000 high-cardinality categorical features (such as postcode and device model) by plug-in mutual information with a fraud label on 20,000 transactions. The top feature, a free-text merchant descriptor with 15,000 distinct values, scored highest. Shuffling the label and recomputing gave almost the same score, so the value was pure estimation bias. Comparing each feature with its own shuffled baseline, and grouping rare values, produced a ranking that held up on new data.

</div>

## A shuffled baseline

The simplest defence works for any estimator: compute the statistic on the real data, then on many copies with one variable shuffled (which destroys dependence while keeping each margin). The real value is only meaningful if it clearly exceeds the shuffled distribution. This is a permutation test, and it also gives a p-value.

## Continuous variables

- Binning: estimates depend on bin width; too fine and the bias explodes, too coarse and dependence is hidden.
- Nearest-neighbour estimators (Kraskov and colleagues): use distances to the $k$-th neighbour in the joint and marginal spaces. They are accurate in low dimensions and widely used for feature screening.
- Neural bounds: train a critic network to maximise a lower bound such as InfoNCE (lesson 15.05). They scale to images and text, but the bounds are either capped (at $\log N$) or have high variance, as Poole and colleagues show. Treat them as training signals, not precise measurements.

## Uncertainty of predictions

For a single input, the entropy of a model's predicted distribution measures how unsure it is. With an ensemble or repeated dropout passes, you get several predictive distributions $p_1, \dots, p_S$ and can split the uncertainty:
$\underbrace{H\big[\tfrac{1}{S}\textstyle\sum_s p_s\big]}_{\text{total}} = \underbrace{\tfrac{1}{S}\textstyle\sum_s H[p_s]}_{\text{expected entropy}} + \underbrace{I(y;\theta \mid x)}_{\text{disagreement}}$.

~~~dg-layers Two sources of predictive uncertainty.
Total predictive entropy | how spread out the averaged prediction is
Expected entropy | uncertainty each member has: noise or ambiguity in the data (aleatoric)
*Mutual information | disagreement between members: the model's own uncertainty (epistemic)
~~~

Active learning methods that pick inputs with high mutual information (often called BALD) target the second kind, which more labels can reduce. Out-of-distribution detection also leans on it: inputs unlike the training data tend to produce disagreement.

## Reporting information quantities

- State units (bits or nats) and the estimator used.
- Report a confidence interval, for example from a bootstrap over samples.
- Compare with a shuffled baseline for mutual information.
- Prefer differences between models evaluated on the same data and estimator; absolute values are less reliable than comparisons.

## Worked numbers

With $N = 100$ samples of a variable that takes $m = 20$ observed values, the Miller-Madow correction adds $(20 - 1)/200 = 0.095$ nats, about 0.14 bits, to the plug-in entropy. If the true distribution is uniform over 20 values, its entropy is $\ln 20 \approx 3.0$ nats, so the correction is small relative to the total but large relative to differences you might want to detect between two features. With $m = 80$ observed values from the same 100 samples, the correction grows to about 0.4 nats and the estimate is unreliable.

<div class="take">
<div class="lab">Key takeaway</div>

information quantities estimated from samples are biased (entropy down, mutual information up), so correct, compare with shuffled baselines and report intervals; for predictions, splitting entropy into expected entropy and mutual information separates data noise from model uncertainty.

</div>
