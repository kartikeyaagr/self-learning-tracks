---
id: "05.05"
course: "machine-learning"
module: "05"
slug: "anomaly-detection"
title: "Anomaly detection"
keyIdeas: ["Anomaly detection gives each point a **score** of how unusual it is, usually without labels, and the scores are used as a ranking. Outlier detection trains on contaminated data; novelty detection trains on clean data and scores new points.", "The z-score rule $|x - \\bar x|/s > 3$ suffers from **masking**: outliers inflate the mean and standard deviation. In a sample of $n$ no z-score can exceed $(n-1)/\\sqrt n$, about 2.85 for $n = 10$. Robust scores use the median and MAD: $0.6745(x - \\text{median})/\\text{MAD} > 3.5$.", "An **isolation forest** cuts on random features at random thresholds; anomalies are alone after few cuts. The score $s = 2^{-E[h(x)]/c(\\psi)}$ is near 1 for anomalies and about 0.5 or below for normal points ($c \\approx 10.2$ for $\\psi = 256$).", "Density methods (kNN distance, LOF, GMM log-likelihood, DBSCAN noise) flag points in sparse regions. **LOF** compares a point's density with its neighbours', so it catches local anomalies next to tight clusters that global methods miss.", "Thresholds are a business decision set by review capacity. Evaluate with the few labels you have using precision at the top $k$ or PR-AUC, because anomalies are rare and accuracy (and often ROC-AUC) looks flattering."]
question: "A payments team flags transactions whose amount has |z| > 3 within each merchant. A merchant with 8 transactions today, one of them a 50× normal amount, is never flagged. Why not, and what would you replace the rule with?"
answer: "The rule is structurally blind, not unlucky. With the sample standard deviation, the largest z-score any point can reach in a sample of $n$ is $(n-1)/\\sqrt n$; for $n = 8$ that is $7/\\sqrt 8 \\approx 2.47$. However extreme the 50× amount is, it inflates the mean and the standard deviation in proportion (masking), so it can never cross 3. Concretely, with seven amounts of 1 and one of 50, the mean is 7.1, the standard deviation 17.3 and the big amount's z-score 2.47, right at the ceiling. Any merchant with 10 or fewer transactions in the window can never be flagged.\n\nReplacement: (1) score against a longer, robust baseline rather than today's handful of points: take the merchant's last 90 days, log-transform amounts (they are skewed, and a 50× jump becomes a shift of about 3.9 in log units), and compute a modified z-score $0.6745(\\log x - \\text{median})/\\text{MAD}$, flagging above about 3.5. The median and MAD cannot be dragged by the outlier itself. (2) For merchants with little history, fall back to the merchant category's distribution or shrink the merchant's median and MAD towards it. (3) Fraud is often unusual only in combination (amount, time of day, country, device, new card), so add a multivariate detector such as an isolation forest on standardised features, and possibly LOF per merchant segment for local anomalies. (4) Do not use a fixed 3 or the library's default contamination: rank by score and set the threshold to the number of cases the review team can handle each day, then check precision at that budget (and PR-AUC) on past labelled fraud, monitoring score distributions over time for drift."
sources: [{"title": "scikit-learn API: IsolationForest", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html", "note": "Parameters (max_samples=256 subsample, contamination) and how score_samples relates to the path-length score; cites Liu, Ting and Zhou (2008)."}, {"title": "scikit-learn User Guide: Novelty and Outlier Detection", "url": "https://scikit-learn.org/stable/modules/outlier_detection.html", "note": "Isolation forest, LOF, robust covariance and One-Class SVM, plus the outlier vs novelty distinction."}, {"title": "Breunig, Kriegel, Ng, Sander (2000): LOF: Identifying Density-Based Local Outliers", "url": "https://www.dbs.ifi.lmu.de/Publikationen/Papers/LOF.pdf", "note": "The local reachability density and why a relative density catches local outliers."}]
practice: ["isolation-forest", "mahalanobis-distance", "normalize-columns", "dbscan"]
tags: ["anomaly detection", "isolation forest", "robust statistics"]
tools: ["NumPy", "scikit-learn"]
---

## What you will learn

- Why detectors output a *score* and a ranking, and outlier versus novelty detection.
- Why the "3 standard deviations" rule can be blind to the outliers it should catch (masking), and how the median and MAD fix it.
- How an isolation forest finds anomalies by counting random cuts, and how its 0-to-1 score is computed.
- How distance and density methods (kNN distance, LOF, GMM, DBSCAN) work, and why LOF catches *local* anomalies.
- How to evaluate a detector when anomalies are rare: precision at your alert budget and PR-AUC.

*Before you start:* means, standard deviations, medians and nearest neighbours. DBSCAN (05.02), Gaussian mixtures (05.03) and PCA (05.04) reappear as detectors; precision and recall are defined below and covered fully in module 06.

## From compressing data to finding what does not fit

The previous lesson (05.04) found a compact description of the bulk of the data. This one asks the opposite question: which points do *not* fit it? That is **anomaly detection**: a fraudulent payment among millions, a failing sensor, a corrupted row in a pipeline.

Labels are rare or missing (tomorrow's fraud may look like nothing seen before), and anomalies are rare by definition, often under 1% of the data, so a model that always says "normal" is 99% accurate and useless. So a detector outputs an **anomaly score** for each point (higher means more unusual), and people use the scores as a **ranking**: investigate the top 100 today, or raise an alert above a threshold chosen to match how many cases the review team can handle.

There are two settings, and it helps to know which one you are in:

- **Outlier detection**: the training data already contains anomalies, and the model of "normal" must be robust to them.
- **Novelty detection**: train on data known to be clean, then score new arrivals. Easier, because normal is not contaminated.

With enough labelled anomalies, a class-weighted supervised classifier (module 06) usually wins; unsupervised detectors matter when anomalies are new, varied or unlabelled.

## The idea in one picture

A temperature sensor sends ten readings. Nine are ordinary; one is a glitch:

| Reading | 8 | 9 | 9 | 10 | 10 | 10 | 11 | 11 | 12 | 100
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

Anyone can see that 100 is wrong. Now try the rule most people reach for first, the **z-score**: how many standard deviations a point sits from the mean. The usual alarm threshold is $|z| > 3$.

~~~dg-fig 05.05-masking.svg
Top: the band "mean plus or minus 3 standard deviations" is so wide (from -66 to 104) that the glitch at 100 sits inside it. Bottom: the robust band built from the median and MAD is narrow, and 100 lies far outside it.
~~~

The mean of the ten readings is 19 and the sample standard deviation is 28.5, so the glitch has $z = (100 - 19)/28.5 = 2.84$. Below 3: no alarm. The glitch has dragged the mean up from about 10 to 19 and inflated the standard deviation from about 1.2 to 28.5, so it has stretched the ruler used to measure it. This is called **masking**: an outlier hides itself (or other outliers) by distorting the statistics.

The fix in the bottom panel replaces the mean with the **median** (the middle value, 10) and the standard deviation with the **median absolute deviation** (MAD, here 1). One wild value cannot move either of them, so 100 stands out by a factor of about 60. That is the whole lesson in miniature: measure "normal" in a way anomalies cannot corrupt, then score each point by its distance from it.

## Step 1: robust scores for a single feature

The classic z-score of a value $x$ is

$$z = \frac{x - \bar x}{s},$$

where $\bar x$ is the sample mean and $s$ the sample standard deviation. For Gaussian data $|z| > 3$ flags about 0.27% of points, but real data has heavy tails (false alarms), and outliers inflate $\bar x$ and $s$ (masking). Masking has a hard limit: with the sample standard deviation, no point in a sample of $n$ values can ever have $|z|$ larger than

$$z_{\max} = \frac{n-1}{\sqrt n},$$

whatever the values are. For $n = 10$ that is $9/\sqrt{10} \approx 2.85$, which is why our glitch got 2.84, almost exactly the ceiling.

~~~dg-fig 05.05-zbound.svg
The largest z-score any point can reach grows only like the square root of the sample size. Up to n = 10 it stays below 3, so a 3-sigma rule applied to small groups can never fire.
~~~

The robust replacement is the **modified z-score**:

$$M = \frac{0.6745\,(x - \tilde x)}{\text{MAD}}, \qquad \text{MAD} = \operatorname{median}_i\,|x_i - \tilde x|,$$

where $\tilde x$ is the median. The constant 0.6745 rescales the MAD so that, for Gaussian data, $M$ is on the same scale as an ordinary z-score. The usual alarm level is $|M| > 3.5$.

<div class="ex">
<div class="lab">Worked example</div>

**the glitch that a 3-sigma rule cannot see**

Readings: 8, 9, 9, 10, 10, 10, 11, 11, 12, 100. Their sum is 190, so the mean is 19.

Sample standard deviation: the squared deviations from 19 add up to 7302 (121 + 100 + 100 + 81 + 81 + 81 + 64 + 64 + 49 + 6561). Divide by $n - 1 = 9$ to get 811.3, and take the square root: $s = 28.5$.

Classic score of the glitch: $z = (100 - 19)/28.5 = 81/28.5 = 2.84$. The ceiling for $n = 10$ is $9/\sqrt{10} = 2.85$, so no value, however absurd, could have crossed 3.

Robust version: sorted, the middle two values are 10 and 10, so the median is 10. Absolute deviations from 10 are 2, 1, 1, 0, 0, 0, 1, 1, 2, 90; their median is 1, so MAD = 1.

Modified score of the glitch: $M = 0.6745 \times (100 - 10)/1 = 60.7$, far above 3.5. Every other reading has $|M| \le 1.35$, so exactly one point is flagged.

Why it works: the median and the MAD still give sensible answers when up to half of the data is contaminated (a **breakdown point** of 50%); the mean and standard deviation can be ruined by a single point (breakdown point 0%).

</div>

Two other robust rules:

- The **IQR fence**: flag values outside $[Q_1 - 1.5\,\mathrm{IQR},\ Q_3 + 1.5\,\mathrm{IQR}]$, where $Q_1$, $Q_3$ are the 25th and 75th percentiles and $\mathrm{IQR} = Q_3 - Q_1$. For our readings that fence is about $[6.6, 13.6]$, which also catches 100. This is the rule behind box-plot whiskers.
- **Percentile thresholds**: flag the top 0.1%. Simple, but it flags something even on a normal day.

For skewed quantities such as payment amounts, apply these rules to $\log x$ rather than $x$, otherwise the long right tail floods you with alarms.

## Step 2: several features at once

One feature at a time misses anomalies that are only strange in *combination*: a 2-metre-tall person is unusual, a 40 kg person is unusual, but a 2-metre, 40 kg person is far stranger than either fact alone suggests. The multivariate version of the z-score is the **Mahalanobis distance**:

$$d_M(x) = \sqrt{(x - \mu)^\top \Sigma^{-1} (x - \mu)},$$

where $\mu$ is the vector of feature means and $\Sigma$ the covariance matrix. Dividing by $\Sigma$ puts each feature on its own scale and accounts for correlation: moving *along* the cloud is cheap, moving *across* it is expensive. Points with equal $d_M$ form an ellipse.

The masking problem comes back in a worse form: outliers inflate $\Sigma$ and stretch the ellipse to include themselves. So estimate $\mu$ and $\Sigma$ robustly, with the **minimum covariance determinant** (MCD) estimator, which fits the covariance on the tightest-packed half of the data. In scikit-learn this is `MinCovDet`, and `EllipticEnvelope` wraps it into a detector. The limit is the shape: it assumes one roughly elliptical cloud, so it fails on data with several clusters or curved structure.

The previous lesson adds one more score: the PCA **reconstruction error**. Points the top components cannot rebuild do not follow the main pattern.

## Step 3: isolation forests, counting random cuts

Isolation forest (Liu, Ting and Zhou, 2008) turns the problem around. Instead of modelling what normal looks like, it measures how *easy* a point is to separate from the rest.

The procedure for one tree, in plain words:

1. Take a random subsample of $\psi$ points (the default is 256).
2. Pick a random feature, and a random threshold between that feature's minimum and maximum in the current group.
3. Split the group in two at that threshold.
4. Repeat inside each half until every point is alone, or a depth limit is reached.

The number of splits needed to reach a point is its **path length** $h(x)$. A random cut often falls between a far-away point and everyone else, so anomalies are alone after few splits; a point in the crowd needs many.

~~~dg-fig 05.05-isolation.svg
One random tree, two targets. The far-away point (left) is alone after 3 cuts; the point in the middle of the cloud (right) needs 12. Each orange line is one random cut.
~~~

One tree is noisy, so the forest averages the path length over many trees (100 by default).

~~~dg-fig 05.05-pathlen.svg
Left: across 400 random trees, the anomaly needs about 3.3 cuts on average and the central point about 11.8. Right: the score turns the average path length into a number between 0 and 1; short paths give scores near 1.
~~~

To make scores comparable across sample sizes, the average path length is divided by $c(\psi)$, the expected path length of a typical point in a tree on $\psi$ points:

$$s(x) = 2^{-E[h(x)]/c(\psi)}, \qquad c(\psi) = 2H(\psi - 1) - \frac{2(\psi - 1)}{\psi},$$

where $E[h(x)]$ is the average path length over the trees and $H(m) = 1 + \tfrac12 + \dots + \tfrac1m$ is the harmonic number (about $\ln m + 0.5772$). The formula for $c$ comes from binary search trees: it is the average depth of an unsuccessful search among $\psi$ keys.

<div class="ex">
<div class="lab">Worked example</div>

**from path length to score**

With $\psi = 256$: $H(255) \approx \ln 255 + 0.5772 = 5.541 + 0.577 = 6.118$. So $c = 2 \times 6.118 - 2 \times 255/256 = 12.237 - 1.992 = 10.24$.

A point isolated in 4 splits on average: $s = 2^{-4/10.24} = 2^{-0.391} = 0.76$. Clearly anomalous.

A point that needs 10 splits: $s = 2^{-10/10.24} = 2^{-0.977} = 0.51$. Borderline.

A point needing exactly $c$ splits scores $2^{-1} = 0.5$. Rule of thumb: scores near 1 are anomalies; if every point scores 0.5 or below, the sample has no clear anomalies.

</div>

~~~dg-flow Isolation forest scoring. A short average path means easy to isolate, which means anomalous.
Point x -(drop down 100 random trees)-> Path lengths h -(average, normalise by c(ψ))-> *Score 2^(−E[h]/c)
~~~

Why it works well: it is linear-time with no distance computations; small subsamples thin out groups of anomalies, reducing masking and **swamping** (normal points near anomalies wrongly flagged); and irrelevant features only dilute it. Weaknesses: axis-parallel cuts create artefacts (Extended Isolation Forest uses slanted cuts), and it is better at global anomalies than at points odd only relative to a tight neighbourhood.

## Step 4: distance and density, and why "local" matters

The other family says anomalies live where data is sparse.

- **kNN distance**: the distance from a point to its $k$-th nearest neighbour. Large distance means a lonely point. Simple and often strong, but it uses one global sense of "far".
- **Local outlier factor (LOF)**: compare a point's local density with the local density of its own neighbours. A score near 1 means "as dense as my neighbours" (normal); well above 1 means "much sparser than my neighbours" (outlier).
- **Density models**: a low log-likelihood under a Gaussian mixture (05.03) or a kernel density estimate.
- **DBSCAN noise**: points that belong to no dense region (05.02) are anomalies by construction.

The difference between global and relative density is easiest to see with two clusters of very different tightness.

~~~dg-fig 05.05-lof.svg
Point A sits just off a very tight cluster. Its distance to its 10th neighbour (about 1.0) is smaller than the gaps inside the loose cluster, so the kNN ranking puts it 23rd. LOF compares A with its own neighbours, which are packed far more tightly, and ranks it first.
~~~

LOF's "local density" is built from **reachability distances**: the distance from $p$ to a neighbour $o$ is taken as $\max(d(p,o),\ k\text{-dist}(o))$, which smooths out tiny distances inside a cluster. The **local reachability density** is one over the average reachability distance to the $k$ neighbours, and

$$\text{LOF}(p) = \frac{\text{average lrd of } p\text{'s neighbours}}{\text{lrd}(p)}.$$

In the figure, A's LOF is about 4: its neighbours are roughly four times denser than A is. A typical point scores about 1.03.

| Method | Good at | Watch out for
| --- | --- | --- |
| Robust z-score / IQR | Single features, fast monitoring | Misses anomalies in combinations of features
| Mahalanobis (robust) | Correlated, roughly Gaussian features | Assumes one elliptical cloud
| Isolation forest | Large, high-dimensional tabular data | Axis artefacts, local anomalies
| LOF / kNN | Local anomalies, clusters of different density | $O(n^2)$ without an index, needs scaling
| GMM / KDE likelihood | Multimodal data with a density story | KDE suffers from the curse of dimensionality

Distance-based methods need **feature scaling** (otherwise grams dominate kilograms), and every method needs a **threshold** or **contamination** setting: a business decision about how many alerts people can review, not something the data decides.

## Anomaly detection in code

scikit-learn gives all detectors the same shape: `fit`, then a score. Below, 980 normal points come from a tight and a loose cluster, and 20 anomalies are scattered uniformly. The labels are used only for evaluation.

```python
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import average_precision_score, roc_auc_score

rng = np.random.default_rng(0)
dense   = rng.normal([0, 0], 0.3, size=(500, 2))      # tight cluster
diffuse = rng.normal([5, 5], 1.5, size=(480, 2))      # loose cluster
odd     = rng.uniform(-4, 10, size=(20, 2))           # 20 scattered anomalies
X = np.vstack([dense, diffuse, odd])
y = np.r_[np.zeros(980), np.ones(20)]                 # 1 = anomaly (evaluation only)

iso = IsolationForest(n_estimators=200, random_state=0).fit(X)
s_iso = -iso.score_samples(X)          # flip sign: this is s(x), higher = stranger

lof = LocalOutlierFactor(n_neighbors=20)
lof.fit_predict(X)
s_lof = -lof.negative_outlier_factor_  # LOF score, about 1 = normal

def precision_at_k(y, s, k):
    return y[np.argsort(-s)[:k]].mean()  # share of real anomalies in the top k

for name, s in [('iforest', s_iso), ('lof', s_lof)]:
    print(name, precision_at_k(y, s, 20), average_precision_score(y, s), roc_auc_score(y, s))
# iforest 0.70  0.73  0.941
# lof     0.75  0.75  0.923
print(s_iso.min(), s_iso.max())         # 0.37 0.79

print((iso.predict(X) == -1).sum())     # 180  <- default threshold flags 18% of the data!

# Novelty detection: fit on clean data, score new arrivals
lof_new = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(X[:980])
print(lof_new.predict([[0, 0], [2, -3]]))   # [ 1 -1]  (1 = normal, -1 = anomaly)
```

Things worth noticing:

- `score_samples` returns the *negative* of the paper's score; flipping the sign gives $s(x)$ in $[0, 1]$.
- With `contamination='auto'`, `predict` cuts at $s = 0.5$, flagging 180 points when only 20 are anomalies. Set `contamination`, or threshold the scores yourself at your alert budget.
- LOF scores its training points by default; `novelty=True` scores new points instead (do not call it on the training set).
- A few of the 20 "anomalies" landed inside a cluster by chance; no detector can find those.

For a single feature the robust score is three lines of NumPy:

```python
x = np.array([8, 9, 9, 10, 10, 10, 11, 11, 12, 100.])
med = np.median(x); mad = np.median(np.abs(x - med))
m = 0.6745 * (x - med) / mad
print(np.where(np.abs(m) > 3.5)[0])     # [9]  (the reading 100)
```

## Checking whether it worked

**Precision** is the share of flagged points that really are anomalies; **recall** is the share of real anomalies that got flagged. With rare anomalies, these are the numbers that matter.

~~~dg-fig 05.05-eval.svg
The isolation forest from the code above. The ROC curve (left) looks nearly perfect at 0.94; the precision-recall curve (right) shows that 30% of the top 20 alerts are false and precision falls fast past 80% recall. A random ranking would score 0.02.
~~~

- **With some labels**, even a few hundred past cases, use ranking metrics: **precision at $k$** (of the top $k$ alerts, how many were real?), recall at your alert budget, or **PR-AUC**. ROC-AUC flatters rare-positive problems because its false-positive rate divides by the huge number of normals: 50 false alarms among 980 normals is only 5%, yet outnumbers the 20 real anomalies.
- **Without labels**, inject synthetic anomalies and check they rise to the top, and have experts review samples of top-scored points.
- **Compare with a dumb baseline**, such as a robust z-score on the most obvious feature; it is often hard to beat.
- **In production**, "normal" drifts, so monitor score distributions (module 09) and re-fit regularly.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Obvious outliers never flagged in small groups | Masking: the 3-sigma ceiling $(n-1)/\sqrt n$ | Median and MAD, or score against a longer history
| Hundreds of alerts per day on skewed metrics | Gaussian rule on heavy-tailed data | Log-transform, robust scores, percentile-based thresholds
| One feature dominates every kNN / LOF ranking | Features on different scales | Standardise or robust-scale features first
| Anomalies next to a dense cluster are missed | Global method (kNN distance, isolation forest) | LOF, or per-segment models
| `predict` flags 10-20% of the data | Default contamination threshold | Set `contamination`, or threshold scores at your review budget
| Great ROC-AUC, team complains about false alarms | ROC hides rare-class precision | Report precision at $k$ and PR-AUC
| Detector quietly degrades over months | Data drift; "normal" changed | Monitor score distributions, scheduled refits

## Going deeper

**The masking bound.** Why can no z-score exceed $(n-1)/\sqrt n$? Put one point at $a$ and the other $n-1$ at $0$; the mean is $a/n$ and the sample variance works out to $a^2/n$, so $z = (a - a/n)/(a/\sqrt n) = (n-1)/\sqrt n$, independent of $a$. Making the point more extreme inflates $s$ in exact proportion. Any other arrangement gives a smaller maximum (this is Samuelson's inequality).

**Why $c(\psi)$ normalises.** Isolation trees have the same structure as binary search trees, and for a random tree on $\psi$ points the expected depth of an external node is $c(\psi) \approx 2\ln\psi$. Dividing by it makes scores comparable across subsample sizes; $s = 0.5$ is exactly "as deep as a typical point". Trees are capped at depth $\lceil \log_2 \psi \rceil$ (8 for 256), because only the shallow part of the tree is needed to separate anomalies; deep paths are completed with the $c(\cdot)$ estimate for the points left in the leaf.

**Robust covariance.** MCD finds the subset of about half the points whose covariance has the smallest determinant (the tightest ellipsoid), via FAST-MCD. With robust $\mu$ and $\Sigma$, squared Mahalanobis distances of clean Gaussian points follow a chi-squared distribution with $d$ degrees of freedom, which gives a principled cut-off such as its 97.5th percentile.

**High dimensions.** There all distances look alike, so kNN and LOF lose contrast; isolation forests or reconstruction errors (PCA, autoencoders) then work better. One-Class SVM learns a kernel boundary around clean data (02.06) for novelty detection.

## Key terms

| Term | Meaning
| --- | --- |
| Anomaly score | A number per point; higher means more unusual. Used as a ranking
| Outlier vs novelty detection | Training data contaminated vs known to be clean
| Masking | Outliers distort the mean and spread so much that they hide themselves
| MAD | Median of absolute deviations from the median; a robust spread
| Modified z-score | $0.6745(x - \text{median})/\text{MAD}$; flag above about 3.5
| Mahalanobis distance | Distance from the centre that accounts for each feature's scale and correlations
| Path length | Number of random cuts needed to isolate a point in an isolation tree
| LOF | Ratio of the neighbours' local density to the point's own; about 1 is normal
| Precision at k | Share of real anomalies among the top $k$ alerts

<div class="take">
<div class="lab">Key takeaway</div>

score points instead of classifying them; measure normal with statistics the anomalies cannot corrupt (median and MAD, robust covariance, small subsamples); reach for isolation forests on large tabular data and LOF for local anomalies; and set thresholds and judge detectors by precision at the alert budget you can actually review.

</div>
