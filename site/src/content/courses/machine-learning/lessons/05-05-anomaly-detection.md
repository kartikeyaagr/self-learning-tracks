---
id: "05.05"
course: "machine-learning"
module: "05"
slug: "anomaly-detection"
title: "Anomaly detection"
keyIdeas: ["Anomaly detection ranks points by how unusual they are, usually without labels. Outlier detection trains on contaminated data; novelty detection trains on clean data and scores new points.", "z-scores flag $\\lvert x-\\bar x\\rvert/s > 3$, but outliers inflate $s$ and mask themselves: in a sample of 10, no point can have a z-score above about 2.85. Robust z-scores use the median and MAD.", "Isolation forests split on random features at random thresholds; anomalies are isolated in few splits. Score $s = 2^{-E[h(x)]/c(\\psi)}$ is near 1 for anomalies and about 0.5 or below for normal points.", "Density methods (kNN distance, LOF, GMM log-likelihood, DBSCAN noise) flag points in low-density regions; LOF compares a point's density with its neighbours', so it catches local anomalies.", "Evaluate with the few labels you have, using precision at the top-$k$ or PR-AUC, because anomalies are rare and accuracy is meaningless."]
question: "A payments team flags transactions whose amount has |z| > 3 within each merchant. A merchant with 8 transactions today, one of them a 50× normal amount, is never flagged. Why not, and what would you replace the rule with?"
answer: "With the sample standard deviation, the largest possible z-score in a sample of $n$ points is $(n-1)/\\sqrt n$. For $n = 8$ that is $7/\\sqrt 8 \\approx 2.47$. However extreme the amount, it inflates both the mean and the standard deviation so much that it can never reach 3, a case of masking. The rule is structurally blind for small groups. I would replace it with a robust score built from the merchant's history rather than just today: a modified z-score using the median and MAD of, say, the last 90 days (log-transformed, since amounts are skewed), flagging values above about 3.5. For merchants with little history, I would fall back to the merchant category's distribution. For a multivariate view (amount, time, location, device), I would add an isolation forest, and I would set thresholds by the review team's capacity, checking precision on past labelled fraud."
sources: [{"title": "scikit-learn API: IsolationForest", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html", "note": "Parameters (max_samples=256 subsample, contamination) and how score_samples relates to the path-length score; cites Liu, Ting and Zhou (2008)."}, {"title": "scikit-learn User Guide: Novelty and Outlier Detection", "url": "https://scikit-learn.org/stable/modules/outlier_detection.html", "note": "Isolation forest, LOF, robust covariance and One-Class SVM, plus the outlier vs novelty distinction."}, {"title": "Breunig, Kriegel, Ng, Sander (2000): LOF: Identifying Density-Based Local Outliers", "url": "https://www.dbs.ifi.lmu.de/Publikationen/Papers/LOF.pdf", "note": "The local reachability density and why a relative density catches local outliers."}]
practice: ["isolation-forest", "mahalanobis-distance", "normalize-columns", "dbscan"]
tags: ["anomaly detection", "isolation forest", "robust statistics"]
tools: ["NumPy", "scikit-learn"]
---

## The problem

An anomaly is a point that does not fit the pattern of the rest: a fraudulent transaction, a failing sensor, a corrupted record. Two traits make the problem unusual. Labels are rare or missing, so most methods are unsupervised and output a score rather than a class. And anomalies are rare by definition, often under 1% of the data, so the output is used as a ranking: investigate the top 100, or alert above a threshold set by review capacity.

There are two settings:

- **Outlier detection**: the training data already contains anomalies, and the model must be robust to them.
- **Novelty detection**: train on data known to be clean, then score new points. Easier, because the model of normal is not contaminated.

If you do have enough labelled anomalies, a supervised classifier with class weighting (module 06, imbalanced data) usually beats everything below. Unsupervised detectors matter when anomalies are new, varied or unlabelled.

## Univariate rules and the masking problem

The z-score $z = (x - \bar x) / s$ with a threshold of 3 flags about 0.27% of a Gaussian. It has two problems. Real data is rarely Gaussian, with heavy tails giving many false alarms. And the outliers themselves inflate $\bar x$ and $s$, hiding themselves. This is called masking. In a sample of $n$ points, no z-score computed with the sample standard deviation can exceed $(n-1)/\sqrt n$, whatever the values.

<div class="ex">
<div class="lab">Worked example</div>

**an outlier that a 3-sigma rule cannot see**

Ten readings: 8, 9, 9, 10, 10, 10, 11, 11, 12, 100. The mean is 19 and the sample standard deviation is 28.5, so the z-score of 100 is 81 / 28.5 = 2.84, below 3. (The bound for n = 10 is 9/√10 ≈ 2.85.)

Robust version: the median is 10, and the median absolute deviation (MAD) is 1. The modified z-score 0.6745 × (100 − 10) / 1 = 60.7 flags it immediately, against the usual cut-off of 3.5.

The median and MAD ignore up to half the data being contaminated; the mean and standard deviation can be ruined by a single point.

</div>

Other robust univariate rules include the IQR fence (outside $[Q_1 - 1.5\,\mathrm{IQR},\ Q_3 + 1.5\,\mathrm{IQR}]$) and percentile thresholds. For several correlated features, the multivariate analogue is the Mahalanobis distance $\sqrt{(x-\mu)^\top \Sigma^{-1} (x-\mu)}$. Use a robust covariance estimate (minimum covariance determinant, `EllipticEnvelope`) for the same masking reason.

## Isolation forests

Isolation forest (Liu, Ting and Zhou, 2008) turns the problem around: instead of modelling normal data, it measures how easy a point is to isolate. Each tree is built on a random subsample of $\psi$ points (default 256) by repeatedly picking a random feature and a random threshold between that feature's min and max, until every point is alone or a depth limit is reached. Anomalies, being few and different, end up alone after very few splits.

~~~dg-flow Isolation forest scoring. Short average path length means easy to isolate, which means anomalous.
Point x -(drop down 100 random trees)-> Path lengths h -(average, normalise by c(ψ))-> *Score 2^(−E[h]/c)
~~~

The score is $s(x) = 2^{-E[h(x)]/c(\psi)}$, where $c(\psi) = 2H(\psi - 1) - 2(\psi-1)/\psi$ is the average path length of an unsuccessful search in a binary search tree, and $H$ is the harmonic number. For $\psi = 256$, $c \approx 10.2$. A point isolated in 4 splits on average scores $2^{-4/10.2} \approx 0.76$; one needing 10 splits scores about 0.51. Scores near 1 are anomalies, and scores at or below 0.5 are normal.

Why it works well in practice: linear time, no distance computations, subsampling protects against masking and swamping, and many irrelevant features only dilute it rather than break it. Weaknesses: axis-parallel splits create artefacts (regions along the axes of a dense cluster look falsely normal; Extended Isolation Forest uses random hyperplanes), and it detects global anomalies better than points that are anomalous only relative to a dense local neighbourhood.

## Density- and distance-based detection

These methods treat low-density regions as anomalous:

- **kNN distance**: the distance to the $k$-th nearest neighbour; simple and strong, but it uses one global scale.
- **Local outlier factor (LOF)**: compares a point's local density with that of its neighbours. A score near 1 is normal and well above 1 is an outlier. Because it is relative, it can flag a point slightly off a tight cluster even when that point is closer to its neighbours than points in a diffuse cluster are to theirs.
- **Density models**: low log-likelihood under a GMM (lesson 05.03) or a kernel density estimate.
- **DBSCAN noise**: points that belong to no dense region (lesson 05.02).

| Method | Good at | Watch out for |
| --- | --- | --- |
| Robust z-score / IQR | Single features, fast monitoring | Misses multivariate anomalies |
| Mahalanobis (robust) | Correlated Gaussian-ish features | One elliptical cloud only |
| Isolation forest | Large, high-dimensional tabular data | Axis artefacts, local anomalies |
| LOF / kNN | Local anomalies, clusters of different density | $O(n^2)$ without an index, needs scaling |
| GMM / KDE likelihood | Multimodal data with a density story | Curse of dimensionality for KDE |

All distance-based methods need sensible feature scaling. All of them also have a contamination or threshold parameter, which is a business decision (how many alerts can be reviewed), not something the data decides.

## Evaluating detectors

If some labels exist, even a few hundred, use ranking metrics: precision at the top $k$ alerts, recall at a fixed alert budget, or PR-AUC. ROC-AUC looks flattering when the positive class is 0.1% of the data. Without labels, inject synthetic anomalies to sanity-check the pipeline and review samples of top-scored points with domain experts. In production, anomaly rates drift with the data itself, so monitor score distributions over time (module 09).

<div class="take">
<div class="lab">Key takeaway</div>

score points instead of classifying them, use robust statistics (median/MAD, robust covariance) so outliers cannot mask themselves, reach for isolation forests on large tabular data and LOF for local anomalies, and evaluate with precision at your alert budget.

</div>
