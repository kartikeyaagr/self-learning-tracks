---
id: "05.01"
course: "machine-learning"
module: "05"
slug: "k-means"
title: "k-means"
keyIdeas: ["Unsupervised learning finds structure without labels: groups (clustering), densities (mixtures, anomaly scores) and low-dimensional coordinates (projections). Every method encodes an assumption about what structure means; for k-means, a cluster is a group of points close to a common centre.", "k-means minimises **inertia** $J = \\sum_i \\lVert x_i - \\mu_{c(i)}\\rVert^2$. Lloyd's algorithm alternates assigning each point to its nearest centre and moving each centre to the mean of its points; neither step can increase $J$.", "Lloyd stops after finitely many passes but only at a local minimum, so the start matters. k-means++ picks each new seed with probability proportional to squared distance from the nearest existing seed, and restarts (`n_init`) keep the best of several runs.", "k-means assumes roughly spherical, similarly sized clusters in a space where Euclidean distance is meaningful: standardise features first, and expect it to fail on elongated, curved or very uneven clusters.", "Choosing $k$ has no single answer: inertia always falls as $k$ grows, so look for an elbow, use the silhouette or gap statistic, or choose the $k$ that serves the downstream task."]
question: "You run k-means with $k=3$ on customers described by `annual_spend` (0-200,000) and `visits_per_month` (0-30). The clusters turn out to be three bands of spend, and visits are ignored entirely. Explain why, and how you would fix it. Would running more restarts help?"
answer: "Inertia is a sum of squared Euclidean distances, so the feature with the largest numeric range dominates. Two customers who differ by 10,000 in spend and 20 in visits are about $10{,}000^2 + 20^2 \\approx 10^8$ apart in squared distance, of which visits contribute 400, i.e. nothing. The cheapest way to lower inertia is therefore to slice the data along spend, and the assignment step barely notices visits at all. More restarts would not help: they find better local minima of the same objective, and this banded result may well be the global minimum of a badly scaled objective. The fix is to make distances meaningful before clustering. Standardise each feature to mean 0 and standard deviation 1 (or use a robust scaler, perhaps after a log transform, if spend is heavily skewed, which it usually is), or deliberately weight features by how much they should matter to the business. After rescaling, I would re-run with k-means++ seeding and `n_init` of 10 or more, check that the clusters are stable across seeds, compare several $k$ with the elbow and silhouette, inverse-transform the centres to describe each segment in original units (for example 'frequent, low-spend' versus 'rare, high-spend'), and finally pick $k$ by whether the marketing team can act on the segments."
sources: [{"title": "Arthur and Vassilvitskii (2007): k-means++: The Advantages of Careful Seeding", "url": "https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf", "note": "The D² seeding rule and its O(log k) guarantee; short and readable."}, {"title": "scikit-learn User Guide: Clustering (K-means)", "url": "https://scikit-learn.org/stable/modules/clustering.html", "note": "Section 2.3.2 on Lloyd, k-means++, mini-batch k-means and the method's assumptions."}, {"title": "Python Data Science Handbook: In Depth: k-Means Clustering", "url": "https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html", "note": "Free chapter with visual examples of where k-means fails."}, {"title": "Tibshirani, Walther, Hastie: Estimating the number of clusters via the gap statistic", "url": "https://hastie.su.domains/Papers/gap.pdf", "note": "The gap statistic for choosing k."}]
practice: ["k-means-clustering", "kmeans-step", "k-means-initialization"]
tags: ["k-means", "k-means++", "inertia"]
tools: ["NumPy", "scikit-learn"]
---

## What you will learn

- What unsupervised learning is, and the four jobs this module covers: clustering, density models, dimensionality reduction and anomaly detection.
- What k-means is trying to minimise (inertia), in words, in symbols and with numbers.
- How Lloyd's algorithm alternates two easy steps, and why it always stops but may stop in the wrong place.
- Why k-means++ seeding and several restarts fix most bad starts.
- When k-means fails (unscaled features, curved or uneven clusters) and how to choose the number of clusters $k$.

*Before you start:* you need Euclidean distance, averages and squared error (the same squared error as least squares in lesson 02.01). Feature scaling from module 03 matters a lot here. No labels and no gradient descent are involved.

## Where this module starts

Modules 01-04 were all **supervised**: every training example came with a label, and the model learned to predict it. This module drops the label. You get $n$ points in $d$ dimensions and nothing else, and the goal is to describe their structure. That is **unsupervised learning**. It comes in four main flavours:

- **Clustering**: which points belong together. k-means here, hierarchical clustering and DBSCAN in 05.02, Gaussian mixtures in 05.03.
- **Dimensionality reduction**: a few new coordinates that keep what matters (05.04).
- **Anomaly detection**: which points do not fit the rest (05.05).
- **Evaluation**: without labels, how can you tell whether any of this is good (05.06)?

With no ground truth to aim at, every method bakes in its own definition of "structure", and that definition decides when it works and when it quietly produces nonsense. k-means has the simplest definition of all: **a cluster is a group of points that sit close to a common centre.**

## The idea in one picture

A bakery delivers to 45 addresses in a town and wants to open 3 small depots. Where should they go? A sensible goal is "every address should be close to its nearest depot". Here are five of the addresses (coordinates in km) and their squared distance to the three depots k-means chose:

| Address (km east, km north) | to depot A (1.73, 1.99) | to depot B (7.78, 3.16) | to depot C (4.66, 8.23) | Served by
| --- | --- | --- | --- | --- |
| (1.3, 1.8) | 0.2 | 43.8 | 52.6 | A
| (3.7, 2.7) | 4.4 | 16.9 | 31.5 | A
| (7.7, 3.2) | 37.1 | 0.0 | 34.5 | B
| (6.5, 4.0) | 26.8 | 2.3 | 21.3 | B
| (5.8, 8.7) | 61.6 | 34.6 | 1.5 | C

Each address goes to the depot with the smallest number in its row. Nobody told the algorithm the neighbourhoods; it found them from the coordinates alone.

~~~dg-fig 05.01-picture.svg
Left: the raw data, 45 addresses with no labels. Right: k-means with k = 3 places three centres (crosses) and colours each address by its nearest centre. The thin lines are the distances k-means tries to keep short.
~~~

Some words you will need:

- A **cluster** is one of the $k$ groups. Here, the addresses served by one depot.
- A **centre** (or **centroid**) $\mu_j$ is the representative point of cluster $j$. In k-means it is always the average of the cluster's points.
- The **assignment** $c(i)$ says which cluster point $i$ belongs to: the one with the nearest centre.
- **Inertia** (within-cluster sum of squares) adds every point's squared distance to its own centre: 92.3 km² here, about 2.05 km² per address, so a typical address is 1.4 km from its depot.

The number $k$ is not learned; you choose it, as the bakery chose three depots.

## Step 1: write down what "good clusters" means

Given $k$, k-means looks for centres $\mu_1, \dots, \mu_k$ and an assignment $c(i)$ for every point that make inertia as small as possible:

$$J = \sum_{i=1}^{n} \lVert x_i - \mu_{c(i)} \rVert^2.$$

Reading each symbol:

- $x_i$ is point $i$, a vector of $d$ features (here $d = 2$: east and north).
- $c(i) \in \{1, \dots, k\}$ is the cluster that point $i$ is assigned to.
- $\mu_{c(i)}$ is the centre of that cluster.
- $\lVert \cdot \rVert^2$ is the squared Euclidean distance: the sum over features of squared differences.

Check: address (3.7, 2.7) with depot A (1.73, 1.99) contributes $(3.7 - 1.73)^2 + (2.7 - 1.99)^2 = 3.88 + 0.50 = 4.4$ to $J$, matching the table.

$J$ is the squared error of regression with each point "predicted" by its centre. It has two tangled kinds of unknowns: which cluster each point is in (discrete) and where the centres are (continuous). The global minimum is NP-hard to find in general, even for $k = 2$, so we split the problem into two easy halves.

## Step 2: the assign step

Suppose someone hands you the centres. What is the best assignment? Each point contributes its own term to $J$ and does not affect anyone else, so each point should simply pick the centre nearest to it:

$$c(i) = \arg\min_j \lVert x_i - \mu_j \rVert^2.$$

That is what the table does row by row; for fixed centres no other assignment gives a smaller $J$.

## Step 3: the update step

Now suppose someone hands you the assignment. Where should each centre go? For one cluster $S_j$ (the set of points assigned to it) we want the point that minimises $\sum_{i \in S_j} \lVert x_i - \mu_j \rVert^2$. Take the derivative with respect to $\mu_j$, set it to zero, and you get

$$\mu_j = \frac{1}{|S_j|} \sum_{i \in S_j} x_i,$$

the **mean** of the cluster's points, where $|S_j|$ is the cluster's size. It is the regression fact "the constant that minimises squared error is the average", and the reason for the name k-*means*.

Number check: if a cluster holds (1, 1), (3, 1) and (2, 4), its centre is $((1+3+2)/3, (1+1+4)/3) = (2, 2)$.

## Step 4: alternate until nothing changes

**Lloyd's algorithm** (the standard k-means algorithm) just repeats the two steps:

1. Start with $k$ initial centres (the **seeds**).
2. **Assign** every point to its nearest centre.
3. **Update** every centre to the mean of its points.
4. If no assignment changed, stop. Otherwise go back to 2.

~~~dg-flow Lloyd's algorithm. Each step can only lower J, and there are finitely many assignments, so it must stop.
Seeds (k-means++) -> Assign to nearest centre -> Move centres to means -(assignments changed?)-> *Converged
~~~

Why must it stop? Neither step can increase $J$, so $J$ never goes up. There are finitely many ways to split $n$ points into $k$ groups and none can be revisited, so it stops, usually within 10 to 100 passes.

<div class="ex">
<div class="lab">Worked example</div>

**Lloyd's algorithm by hand on six points**

Points: P1 = (1, 1), P2 = (3, 1), P3 = (2, 4), P4 = (6, 5), P5 = (8, 5), P6 = (7, 8), with k = 2. Deliberately bad seeds: $\mu_1$ = P1 = (1, 1) and $\mu_2$ = P3 = (2, 4), both in the lower-left group.

Pass 1, assign. P2: squared distance to $\mu_1$ is $2^2 + 0^2 = 4$, to $\mu_2$ is $1^2 + 3^2 = 10$, so cluster 1. P4: $5^2 + 4^2 = 41$ versus $4^2 + 1^2 = 17$, so cluster 2. P5: $49 + 16 = 65$ versus $36 + 1 = 37$, cluster 2. P6: $36 + 49 = 85$ versus $25 + 16 = 41$, cluster 2. Clusters: {P1, P2} and {P3, P4, P5, P6}. Inertia: $0 + 4 + 0 + 17 + 37 + 41 = 99$.

Pass 1, update. $\mu_1 = ((1+3)/2, (1+1)/2) = (2, 1)$. $\mu_2 = ((2+6+8+7)/4, (4+5+5+8)/4) = (5.75, 5.5)$. Inertia with the same assignment but new centres: cluster 1 gives $1 + 1 = 2$; cluster 2 gives P3 $14.0625 + 2.25 = 16.3125$, P4 $0.0625 + 0.25 = 0.3125$, P5 $5.0625 + 0.25 = 5.3125$, P6 $1.5625 + 6.25 = 7.8125$. Total $J = 31.75$, already far below 99.

Pass 2, assign. P3 = (2, 4) is now $0^2 + 3^2 = 9$ from $\mu_1$ but $16.3125$ from $\mu_2$, so it switches to cluster 1. Every other point stays. Clusters: {P1, P2, P3} and {P4, P5, P6}. Inertia: $1 + 1 + 9 + 0.3125 + 5.3125 + 7.8125 = 24.4375$.

Pass 2, update. $\mu_1 = (2, 2)$ and $\mu_2 = ((6+8+7)/3, (5+5+8)/3) = (7, 6)$. Inertia: cluster 1 gives $2 + 2 + 4 = 8$, cluster 2 gives $2 + 2 + 4 = 8$, total $J = 16$.

Pass 3, assign. Nothing changes (for example P3 is 4 from $\mu_1$ and 29 from $\mu_2$), so the algorithm stops. The sequence of inertias was 99, 31.75, 24.44, 16: it only ever went down.

</div>

~~~dg-fig 05.01-lloyd.svg
The hand example, pass by pass. Filled crosses are the current centres, hollow crosses where the update step moves them. In pass 2 the point P3 changes colour, and by pass 3 nothing moves.
~~~

## Step 5: the start matters, so seed with k-means++

"Never goes up" is not "finds the best answer". Lloyd reaches a **local minimum**: a clustering no assign or update step can improve, yet possibly much worse than the best. Centres only move toward nearby points, so if two seeds start in one true group and none in another, Lloyd rarely recovers.

~~~dg-fig 05.01-inertia.svg
Lloyd's algorithm started from eight random seeds on data with six clear blobs. Every curve only goes down, but four of them flatten out at three to seven times the best inertia: stuck in a worse local minimum.
~~~

The fix is to choose seeds that are spread out. **k-means++** (Arthur and Vassilvitskii, 2007) does it with one rule:

1. Pick the first centre uniformly at random from the data.
2. For every point, compute $D(x)$, its distance to the nearest centre chosen so far.
3. Pick the next centre at random, with probability proportional to $D(x)^2$. Points far from all existing centres are much more likely to be picked.
4. Repeat steps 2-3 until there are $k$ centres, then run Lloyd as usual.

<div class="ex">
<div class="lab">Worked example</div>

**why squared-distance seeding finds the other cluster**

One-dimensional data: 1, 2, 3, 10, 11, 12, with k = 2. Suppose the first seed is 1.

Squared distances to it: 2 → 1, 3 → 4, 10 → 81, 11 → 100, 12 → 121. Total 307.

The chance the second seed lands in {10, 11, 12} is $(81 + 100 + 121)/307 = 302/307 \approx 98.4\%$. Uniform seeding among the other five points would give 3/5 = 60%.

From seeds 1 and 11, one Lloyd pass assigns {1, 2, 3} and {10, 11, 12}, giving centres 2 and 11, the optimum, with inertia $1 + 0 + 1 + 1 + 0 + 1 = 4$.

</div>

Libraries add **restarts** (`n_init` in scikit-learn): run several times from different seeds and keep the lowest inertia.

~~~dg-fig 05.01-init.svg
The same six-blob data. Left: a random start crowds three centres into one blob, while two centres sit in the gaps and each covers two blobs, inertia 765. Right: k-means++ seeding puts one centre in each blob, inertia 118.
~~~

## k-means in code

In practice, scale first, then call scikit-learn:

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=600, centers=4, cluster_std=1.2, random_state=42)
Xs = StandardScaler().fit_transform(X)           # equal footing for every feature

km = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=0).fit(Xs)
print(km.inertia_)                    # 50.89  (k = 1 would give 1200 = n * d)
print(km.n_iter_)                     # 2      Lloyd passes in the best run
print(np.bincount(km.labels_))        # [150 149 150 151]
print(km.cluster_centers_.shape)      # (4, 2)

for k in range(2, 8):                 # compare several k
    m = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
    print(k, round(m.inertia_, 1), round(silhouette_score(Xs, m.labels_), 3))
# 2 638.5 0.544
# 3 156.8 0.72
# 4 50.9 0.753     <- inertia stops dropping fast, silhouette peaks
# 5 45.9 0.642
# 6 41.3 0.512
# 7 36.7 0.411
```

`labels_` holds $c(i)$, `predict` assigns new points to the nearest centre, and `inertia_` is $J$. Cluster numbers are arbitrary: another seed may rename "cluster 0" to "cluster 2". Recent scikit-learn defaults to `n_init="auto"` (one run with k-means++); set it explicitly for restarts.

Writing Lloyd yourself takes a few lines of NumPy and is worth doing once:

```python
def lloyd(X, C, iters=100):
    for _ in range(iters):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)   # (n, k) squared distances
        a = d2.argmin(1)                                      # assign step
        newC = np.array([X[a == j].mean(0) if np.any(a == j) else C[j]
                         for j in range(len(C))])             # update step
        if np.allclose(newC, C): break
        C = newC
    return C, a, d2[np.arange(len(X)), a].sum()               # centres, labels, inertia

P = np.array([[1, 1], [3, 1], [2, 4], [6, 5], [8, 5], [7, 8.]])
print(lloyd(P, P[[0, 2]]))   # centres [[2, 2], [7, 6]], labels [0 0 0 1 1 1], inertia 16.0
```

The `if np.any(a == j)` guard handles an **empty cluster** (a centre that no point chose); libraries instead re-seed such a centre at a far-away point. Each pass costs $O(nkd)$: every point against every centre in every dimension. For millions of points, `MiniBatchKMeans` updates centres from small random batches and is many times faster for a small increase in inertia.

## Choosing k

Inertia cannot pick $k$ by itself, because it always falls as $k$ grows: an extra centre can only bring some points closer. At $k = n$ every point is its own centre and $J = 0$. For the delivery data, inertia goes 697 (k = 1), 358, 92 (k = 3), 75, 60, 51 and so on. Common tools:

- **Elbow**: plot inertia against $k$ and look for the bend where extra clusters stop paying off.
- **Silhouette**: compare each point's mean distance to its own cluster ($a$) with that to the nearest other cluster ($b$); $(b - a)/\max(a, b)$ near 1 means well placed. Pick the $k$ with the highest average (05.06).
- **Gap statistic**: compare $\log J_k$ with what you would get on structureless uniform data of the same extent, and pick the smallest $k$ where the gap stops growing meaningfully.
- **The task**: if clusters feed a product (segments a team acts on, an image codebook), choose the $k$ that serves it.

~~~dg-fig 05.01-choosek.svg
The delivery data. Left: inertia keeps falling, but the drop flattens sharply after k = 3. Right: the mean silhouette is highest (0.69) at k = 3.
~~~

## Reading and checking the result

k-means always returns $k$ clusters, even on structureless data, so check before trusting it:

- **Rerun with other seeds.** Clusters that change a lot mean weak structure or a wrong $k$.
- **Look at cluster sizes.** One giant cluster plus tiny ones often means outliers or bad scaling.
- **Describe each centre in original units.** Inverse-transform the centres: "frequent, low-spend customers" is a usable segment.
- **Compare inertia with $k = 1$.** $1 - J_k/J_1$ is the fraction of total spread explained by the clustering. For the depots it is $1 - 92.3/697.1 \approx 87\%$.

## What goes wrong

k-means draws straight boundaries halfway between centres and scores clusters by squared Euclidean distance. Every failure below follows from that.

~~~dg-fig 05.01-failures.svg
Left: spend in currency swamps visits per month, so k-means splits on spend and ignores the two real groups (low and high visits). Middle: two interleaved curves are cut by a straight line. Right: a wide cluster of 250 points is halved, while the tight cluster of 25 is merged into one half.
~~~

| Problem | Symptom | Fix
| --- | --- | --- |
| Unscaled features | Clusters are bands along the feature with the biggest range | Standardise; log-transform skewed features; weight features on purpose
| Elongated, curved or nested shapes | Natural groups cut in half or glued together | DBSCAN or spectral clustering (05.02), full-covariance mixture (05.03)
| Very different sizes or spreads | Big cluster split, small tight one absorbed | Gaussian mixtures (05.03); check sizes
| Outliers | A centre dragged toward or captured by a few extreme points | Remove them first (05.05) or use k-medoids
| Bad start | Different results per run, high inertia | k-means++ and `n_init` of 10 or more
| Many dimensions | Distances all look alike; clusters are noise | Reduce with PCA first (05.04)

The left panel is the commonest real mistake, and restarts cannot fix it: they find better minima of an objective that is itself dominated by the biggest-numbered feature.

## Going deeper

**Voronoi geometry.** For fixed centres, the set of points nearest to $\mu_j$ is a convex polygon (a Voronoi cell), and the boundary between two clusters is the perpendicular bisector of their centres. That is why k-means can only produce convex clusters separated by straight lines.

**Inertia as pairwise spread.** A short calculation gives $\sum_{i \in S_j} \lVert x_i - \mu_j \rVert^2 = \frac{1}{2|S_j|} \sum_{i, i' \in S_j} \lVert x_i - x_{i'} \rVert^2$. So k-means also minimises squared distances between points of the same cluster. Since total spread = within-cluster + between-cluster, lowering inertia equals pushing centres apart.

**A probabilistic view.** k-means is the limit of a Gaussian mixture in which every component has the same spherical covariance $\sigma^2 I$ and $\sigma \to 0$, so soft responsibilities become hard assignments. Lloyd's two steps are then the E-step and M-step of EM. Lesson 05.03 makes this exact and shows what you gain by letting each cluster have its own shape and size.

**Guarantees and speed.** k-means++ alone gives an expected inertia within a factor $O(\log k)$ of the optimum (the bound is $8(\ln k + 2)$), before Lloyd improves it further. Lloyd's worst case needs exponentially many passes, yet it is fast in practice (smoothed analysis explains why). Elkan's variant uses the triangle inequality to skip most distance computations.

**Relatives.** **k-medoids** uses actual data points as centres and any distance, so it resists outliers. **Vector quantisation** replaces each vector by the index of its nearest k-means centre, the basis of colour quantisation and of many approximate nearest-neighbour indexes.

## Key terms

| Term | Meaning
| --- | --- |
| Unsupervised learning | Finding structure in data that has no labels
| Cluster | A group of points that the method treats as belonging together
| Centre / centroid | The mean of a cluster's points; the cluster's representative
| Inertia | Sum of squared distances from each point to its own centre (within-cluster sum of squares)
| Lloyd's algorithm | Alternate assigning points to the nearest centre and moving centres to their means
| Local minimum | A clustering no single step can improve, possibly worse than the best one
| k-means++ | Seeding that picks each new centre with probability proportional to squared distance from existing ones
| Restarts (`n_init`) | Several full runs from different seeds, keeping the lowest inertia
| Elbow / silhouette | Plots or scores used to choose the number of clusters

<div class="take">
<div class="lab">Key takeaway</div>

k-means alternates nearest-centre assignment and mean updates to reach a local minimum of inertia; seed with k-means++, run several restarts, standardise features, and choose $k$ with a criterion other than raw inertia.

</div>
