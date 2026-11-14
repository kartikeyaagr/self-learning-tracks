---
id: "05.01"
course: "machine-learning"
module: "05"
slug: "k-means"
title: "k-means"
keyIdeas: ["Unsupervised learning finds structure without labels: groups (clustering), densities (mixtures, anomaly scores) and low-dimensional coordinates (projections). Every method encodes an assumption about what structure means.", "k-means minimises inertia $\\sum_i \\lVert x_i - \\mu_{c(i)}\\rVert^2$. Lloyd's algorithm alternates assigning points to the nearest centre and moving each centre to its cluster mean; neither step can increase inertia.", "Lloyd converges in finitely many steps but only to a local minimum, so initialisation matters. k-means++ picks each new seed with probability proportional to squared distance from the nearest existing seed.", "k-means assumes roughly spherical, similarly sized clusters in a space where Euclidean distance is meaningful: standardise features first, and expect it to fail on elongated or nested shapes.", "Choosing $k$ has no single answer: inertia always falls with $k$, so look for an elbow, use silhouette or the gap statistic, or pick $k$ by the downstream task."]
question: "You run k-means with $k=3$ on customers described by `annual_spend` (0-200,000) and `visits_per_month` (0-30). The clusters turn out to be three bands of spend, and visits are ignored entirely. Explain why, and how you would fix it. Would running more restarts help?"
answer: "Inertia is a sum of squared Euclidean distances, so the feature with the largest numeric range dominates. Differences in spend are thousands of times larger than differences in visits, so the cheapest way to lower inertia is to slice the data along spend; visits barely change any distance. More restarts would not help: they find better local minima of the same objective, and this result may well be the global minimum of a badly scaled objective. The fix is to make distances meaningful: standardise each feature (or use a robust scaler if spend is heavily skewed, perhaps after a log transform), or weight features by how much they should matter to the business. After rescaling, I would re-run with k-means++ seeding and several restarts, and compare $k$ values with silhouette scores and with whether the segments are useful."
sources: [{"title": "Arthur and Vassilvitskii (2007): k-means++: The Advantages of Careful Seeding", "url": "https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf", "note": "The D² seeding rule and its O(log k) guarantee; short and readable."}, {"title": "scikit-learn User Guide: Clustering (K-means)", "url": "https://scikit-learn.org/stable/modules/clustering.html", "note": "Section 2.3.2 on Lloyd, k-means++, mini-batch k-means and the method's assumptions."}, {"title": "Python Data Science Handbook: In Depth: k-Means Clustering", "url": "https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html", "note": "Free chapter with visual examples of where k-means fails."}, {"title": "Tibshirani, Walther, Hastie: Estimating the number of clusters via the gap statistic", "url": "https://hastie.su.domains/Papers/gap.pdf", "note": "The gap statistic for choosing k."}]
practice: ["k-means-clustering", "kmeans-step", "k-means-initialization"]
tags: ["k-means", "k-means++", "inertia"]
tools: ["NumPy", "scikit-learn"]
---

## Unsupervised learning in one page

Modules 01-04 learned a mapping from features to a label. This module drops the label. The data is just $n$ points in $d$ dimensions, and the goal is to describe its structure:

- **Clustering**: which points belong together (k-means here, hierarchical and DBSCAN in 05.02, mixtures in 05.03).
- **Dimensionality reduction**: a few coordinates that keep what matters (05.04).
- **Anomaly detection**: which points do not fit (05.05).
- **Evaluation**: without labels, how do you know any of it is good (05.06)?

There is no ground truth to optimise, so every method bakes in a definition of structure. k-means says a cluster is a set of points close to a common centre.

## The objective and Lloyd's algorithm

Given $k$, k-means seeks centres $\mu_1, \dots, \mu_k$ and an assignment $c(i)$ of each point that minimise the within-cluster sum of squares, called inertia:

$$J = \sum_{i=1}^{n} \lVert x_i - \mu_{c(i)} \rVert^2$$

Finding the global minimum is NP-hard even for $k = 2$ in general dimension. Lloyd's algorithm finds a local minimum by alternating two easy steps:

1. **Assign**: put each point in the cluster of its nearest centre. For fixed centres this is the best possible assignment.
2. **Update**: move each centre to the mean of its points. For a fixed assignment the mean minimises squared distance.

~~~dg-flow Lloyd's algorithm. Each step can only lower J, and there are finitely many assignments, so it must stop.
Seeds (k-means++) -> Assign to nearest centre -> Move centres to means -(assignments changed?)-> *Converged
~~~

Since neither step can increase $J$ and there are finitely many partitions, the algorithm terminates, usually in 10-100 iterations. Each iteration costs $O(nkd)$. For large $n$, mini-batch k-means updates centres from small random batches and is far faster for a small loss in inertia.

```python
import numpy as np
def lloyd(X, C, iters=100):
    for _ in range(iters):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)   # (n, k) squared distances
        a = d2.argmin(1)
        newC = np.array([X[a == j].mean(0) if np.any(a == j) else C[j] for j in range(len(C))])
        if np.allclose(newC, C): break
        C = newC
    return C, a, d2[np.arange(len(X)), a].sum()            # centres, labels, inertia
```

The `if np.any(a == j)` guard handles empty clusters; libraries instead re-seed an empty centre at a far-away point.

## Seeding with k-means++

Lloyd's local minimum depends heavily on the starting centres. Random points from the data often put two seeds in one true cluster and none in another, and Lloyd cannot fix that because centres move only locally. k-means++ (Arthur and Vassilvitskii, 2007) spreads the seeds out:

1. Pick the first centre uniformly at random from the data.
2. For every point, let $D(x)$ be the distance to the nearest centre chosen so far.
3. Pick the next centre with probability proportional to $D(x)^2$. Repeat until there are $k$.

This seeding alone is $O(\log k)$-competitive with the optimal inertia in expectation, and in practice it also speeds Lloyd's convergence. Libraries add restarts too (`n_init`): run several seedings and keep the lowest inertia.

<div class="ex">
<div class="lab">Worked example</div>

**why squared-distance seeding finds the other cluster**

One-dimensional data: 1, 2, 3, 10, 11, 12, with k = 2. Suppose the first seed is 1.

Squared distances to it: 2 → 1, 3 → 4, 10 → 81, 11 → 100, 12 → 121. Total 307.

The chance the second seed lands in {10, 11, 12} is 302 / 307 ≈ 98.4%. Uniform seeding would give 3/5 = 60%.

From seeds 1 and 11, one Lloyd pass gives centres 2 and 11, the optimum, with inertia 4.

</div>

## What k-means assumes

k-means is equivalent to a Gaussian mixture with equal, spherical covariances and hard assignments (lesson 05.03 makes this precise). Its failures follow from that:

- **Scale**: distances mix units, so a feature in dollars swamps one in years. Standardise or otherwise choose a meaningful metric first.
- **Shape**: boundaries between clusters are straight lines (a Voronoi diagram), so elongated, curved or nested clusters get split or merged. Use DBSCAN, spectral clustering or a full-covariance GMM.
- **Size and density**: inertia favours splitting a large, spread cluster over keeping a small tight one separate.
- **Outliers**: means are pulled by extreme points; k-medoids (centres must be data points, any distance) is more robust.
- **Dimensionality**: in hundreds of dimensions distances concentrate; reduce with PCA first (05.04).

## Choosing k

Inertia always decreases as $k$ grows (reaching 0 at $k = n$), so it cannot pick $k$ by itself. Common tools:

- **Elbow**: plot inertia against $k$ and look for the bend where gains flatten. Often there is no clear bend.
- **Silhouette**: average how much closer each point is to its own cluster than to the next one, and choose the $k$ with the highest score (05.06).
- **Gap statistic**: compare $\log J_k$ with its expected value on uniform reference data, and pick the smallest $k$ where the gap stops growing meaningfully.
- **The task**: if clusters feed a product (customer segments, a codebook for vector quantisation), choose the $k$ that serves it: segments a team can act on, or the compression-quality trade-off.

<div class="take">
<div class="lab">Key takeaway</div>

k-means alternates nearest-centre assignment and mean updates to reach a local minimum of inertia; seed with k-means++, run several restarts, standardise features, and choose $k$ with a criterion other than raw inertia.

</div>
