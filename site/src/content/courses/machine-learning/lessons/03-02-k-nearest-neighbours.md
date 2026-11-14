---
id: "03.02"
course: "machine-learning"
module: "03"
slug: "k-nearest-neighbours"
title: "k-nearest neighbours"
keyIdeas: ["kNN stores the training set and predicts by majority vote (or mean) of the $k$ closest points: no training cost, $O(nd)$ work per query by brute force.", "The distance metric *is* the model: unscaled features let the largest-unit feature dominate, and irrelevant features add noise to every distance.", "Small $k$ means low bias and high variance (1-NN has zero training error); large $k$ smooths toward the majority class. Choose $k$ by cross-validation.", "In high dimensions neighbourhoods stop being local: to capture 1% of uniform data in 10 dimensions you need a cube spanning 63% of each axis, and distances concentrate.", "kNN works when data lie near a low-dimensional manifold or in a learned embedding space; at scale it relies on tree indexes or approximate nearest-neighbour search."]
question: "A kNN classifier gets 90% accuracy on a 5-feature problem. You append 50 features of pure random noise (same scale as the standardised real features). Predict what happens to kNN's accuracy and to an L1-regularised logistic regression's accuracy, and explain the difference."
answer: "kNN degrades sharply, likely toward chance. Squared Euclidean distance is a sum over features, so the 50 noise features contribute about 10 times as much variation as the 5 real ones. Neighbours are then chosen mostly by noise, and the vote is close to random among nearby labels. Distances also concentrate as dimension rises, so the nearest points are barely nearer than the rest. kNN has no mechanism to discount a feature. L1-regularised logistic regression barely changes. Each noise feature has near-zero correlation with the label, so its coefficient falls below the soft-threshold and is set exactly to zero, leaving essentially the original 5-feature model. There is a small variance cost from occasional spurious selections. The lesson: models that learn per-feature weights can ignore junk, while kNN needs feature selection or a learned metric done for it."
sources: [{"title": "Cornell CS4780 lecture notes: k-Nearest Neighbors", "url": "https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote02_kNN.html", "note": "The algorithm, the Cover-Hart bound and a clear treatment of the curse of dimensionality."}, {"title": "scikit-learn: Nearest Neighbors", "url": "https://scikit-learn.org/stable/modules/neighbors.html", "note": "Brute force vs KD-tree vs ball tree, when each wins, and the available metrics."}, {"title": "The Elements of Statistical Learning, ch. 2.5 and 13", "url": "https://hastie.su.domains/ElemStatLearn/", "note": "Local methods in high dimensions (the edge-length calculation) and prototype methods."}]
practice: ["knn-classifier", "distance-metrics", "euclidean-distance", "feature-scaling", "embedding-similarity"]
tags: ["distance metrics", "curse of dimensionality"]
tools: ["NumPy", "scikit-learn"]
---

## The algorithm

k-nearest neighbours is the purest instance-based method. "Training" is storing the data. To predict for a query $x$:

1. Compute its distance to every training point.
2. Take the $k$ closest.
3. Classification: majority vote among their labels, optionally weighted by $1/\text{distance}$. Regression: their mean.

There is no fitted parameter vector, so kNN can represent arbitrarily complicated boundaries given enough data. A classic result (Cover and Hart, 1967) says that as $n \to \infty$ the 1-NN error is at most twice the Bayes error, which is remarkable for a method with no training. The catch is in "as $n \to \infty$": the data needed grows quickly with dimension.

~~~dg-flow Every prediction scans the stored data; the metric and k are the only modelling choices.
Query x -(distance to all n points)-> Sorted distances -(take k smallest)-> *k neighbours -(vote or average)-> Prediction
~~~

## The distance metric is the model

kNN has no weights to learn which features matter, so the metric decides.

| Metric | Formula | Use when
| --- | --- | --- |
| Euclidean | $\sqrt{\sum_j (x_j - x'_j)^2}$ | continuous, standardised features
| Manhattan | $\sum_j |x_j - x'_j|$ | robust to a single large coordinate difference
| Cosine distance | $1 - \frac{x^\top x'}{\|x\|\|x'\|}$ | direction matters, length doesn't: text, embeddings
| Mahalanobis | $\sqrt{(x - x')^\top S^{-1}(x - x')}$ | correlated features with a known covariance
| Hamming | number of differing coordinates | categorical or binary codes

Two failure modes come up constantly:

- **Scale.** With income in dollars (spread around 30,000) and age in years (spread around 12), Euclidean distance is essentially income distance. Standardise with training statistics, and apply the same transform to queries.
- **Irrelevant features.** Each useless feature adds random variation to every distance. With 5 useful and 50 noise features, nearest neighbours are mostly chosen by noise. Feature selection or a learned metric matters more for kNN than for any model in module 02, which can set an irrelevant weight to zero.

## Choosing k

$k$ is the complexity knob, running in the opposite direction from most knobs. With $k = 1$ every training point is its own neighbour: zero training error, a jagged boundary that wraps around every noisy label, high variance. As $k$ grows, predictions average over more points and the boundary smooths. At $k = n$ every query gets the majority class. Validation error against $k$ is U-shaped, as lesson 01.02 predicts, and a rough sense of effective complexity is $n/k$ "parameters".

Practical rules: choose $k$ by cross-validation over values like 1, 3, 5, ..., 51; prefer odd $k$ for binary problems to avoid ties; use distance weighting when the closest neighbours are much closer than the $k$-th; and expect larger optimal $k$ with noisier labels.

## The curse of dimensionality

kNN assumes your neighbours are similar to you. In high dimensions there are no close neighbours.

**Volume.** Put $n$ points uniformly in the unit cube $[0,1]^d$. A sub-cube capturing a fraction $r$ of the points needs edge length $r^{1/d}$. For $r = 0.01$: 0.1 in 2 dimensions, 0.63 in 10, 0.955 in 100. A neighbourhood holding 1% of the data spans almost the whole range of every feature, so it is not local any more.

**Distance concentration.** For many independent features, the distances from a query to its nearest and farthest points become nearly equal: the ratio $(\text{max} - \text{min})/\text{min}$ shrinks toward 0 as $d$ grows. When everything is about equally far away, "nearest" carries little information.

**Sample size.** To keep the same neighbourhood size as $d$ grows, $n$ must grow exponentially in $d$.

kNN still works in practice for two reasons. Real data usually lie near a low-dimensional manifold: images of digits have 784 pixels but far fewer real degrees of freedom. And you can move to a space where distance means something: PCA (module 05) or a learned embedding, where kNN becomes retrieval.

## Making it fast

Brute force is $O(nd)$ per query, which is fine for $10^5$ points and vectorises well:

```python
import numpy as np
def knn_predict(Xtr, ytr, Xq, k=5):           # ytr: non-negative int labels
    d2 = (Xq**2).sum(1)[:, None] + (Xtr**2).sum(1)[None, :] - 2 * Xq @ Xtr.T
    idx = np.argpartition(d2, k, axis=1)[:, :k]   # k smallest, unordered: O(n) per row
    return np.array([np.bincount(row).argmax() for row in ytr[idx]])
```

KD-trees and ball trees give exact answers faster in low dimensions (roughly $d \lesssim 20$) but degrade to brute force beyond that, the curse again. For millions of embedding vectors, use approximate nearest-neighbour indexes such as HNSW or IVF, which trade a little recall for orders-of-magnitude speed; the AI Engineering course (lessons 02.06-02.07) covers them.

<div class="ex">
<div class="lab">Worked example</div>

**k changes the answer**

Red points at (1, 1) and (2, 1); blue points at (3, 2.2), (4, 3) and (5, 4). Query (2.5, 1.5).

Distances: (2, 1) 0.71 red, (3, 2.2) 0.86 blue, (1, 1) 1.58 red, (4, 3) 2.12 blue, (5, 4) 3.54 blue.

$k = 1$: red. $k = 3$: red 2, blue 1, so red. $k = 5$: blue 3, red 2, so blue. Multiply the second coordinate by 100 (a unit change) and the neighbour order is set almost entirely by that coordinate: the answer depends on units, not on the problem.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

kNN is only as good as its distance. Standardise, drop irrelevant features, pick k by cross-validation, and in high dimensions move to a space (PCA or a learned embedding) where nearness means similarity.

</div>
