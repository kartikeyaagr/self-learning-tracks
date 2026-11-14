---
id: "05.06"
course: "machine-learning"
module: "05"
slug: "evaluating-clusters"
title: "Evaluating clusters"
keyIdeas: ["Cluster quality can be judged three ways: **internal** indices that use only the data and labels (inertia, silhouette), **external** indices against known classes (ARI, NMI), and usefulness for the task the clusters serve.", "Inertia always falls with $k$ and only suits comparisons at fixed $k$. The **silhouette** $s = (b-a)/\\max(a,b)$ compares each point's mean distance to its own cluster ($a$) and to the nearest other cluster ($b$); negative values flag likely misassignments.", "Internal indices encode a definition of good clusters, usually compact and convex, so they penalise correct non-convex clusterings such as DBSCAN's half-moons.", "External indices must ignore label names and be corrected for chance: purity is trivially 1 with one cluster per point, while ARI and AMI are about 0 for random labels.", "Check stability: re-cluster bootstrap samples or other seeds and compare runs with ARI. Structure that changes between runs is not structure."]
question: "Two clusterings of the same 10,000 documents, compared with 20 known topic labels: A is k-means with k=20 (purity 0.62, NMI 0.48, ARI 0.31); B is a k-means run with k=500 (purity 0.81, NMI 0.58, ARI 0.09). A colleague prefers B because two of the three numbers are higher. Who is right, and what would you report or check to settle it?"
answer: "A is the better match to the topics, and the colleague is being misled by two scores that are not corrected for chance.\n\n**Why purity and NMI favour B.** Purity counts, for each cluster, the share of its majority topic. With 500 clusters of about 20 documents each, most tiny clusters are dominated by one topic almost by accident, so purity rises; in the limit of one document per cluster it is exactly 1. NMI's normalisation is also known to drift upward when the number of clusters is large relative to $n$: even random labels with 500 clusters would get a clearly positive NMI. Neither number says B captured the topics better; they mostly reward fragmentation.\n\n**Why ARI favours A.** ARI counts document pairs that the clustering and the topic labels treat the same way, then subtracts what random labellings with the same group sizes would achieve. Splitting each topic into about 25 fragments breaks most same-topic pairs apart, so the pair agreement collapses and ARI falls to 0.09, close to the chance level of 0. A keeps far more same-topic pairs together (0.31).\n\n**What I would report.** AMI (NMI corrected for chance) alongside ARI; both should favour A. Homogeneity and completeness would show the trade-off directly: B is highly homogeneous (each cluster is one topic) but has very low completeness (each topic is scattered across many clusters). I would also check stability of each solution across seeds with ARI, since k = 500 on text is likely to move a lot between runs.\n\n**The real question.** If fine-grained clusters are actually wanted, for example as retrieval buckets or as features for a model, then agreement with 20 topics is the wrong yardstick. Evaluate B on that task (retrieval quality, validation score of the downstream model) instead of by its match to the topic labels."
sources: [{"title": "scikit-learn User Guide: Clustering performance evaluation", "url": "https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation", "note": "Definitions, pros and cons of ARI, AMI/NMI, homogeneity, silhouette, Calinski-Harabasz and Davies-Bouldin."}, {"title": "scikit-learn example: Selecting the number of clusters with silhouette analysis", "url": "https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html", "note": "Silhouette plots for several k, showing how to read weak clusters."}, {"title": "Vinh, Epps, Bailey (2010): Information Theoretic Measures for Clusterings Comparison", "url": "https://jmlr.org/papers/v11/vinh10a.html", "note": "Why NMI needs a chance adjustment, and the case for AMI."}]
practice: ["k-means-clustering", "agglomerative-clustering", "gaussian-mixture-models-em", "dbscan"]
tags: ["silhouette", "adjusted Rand index", "cluster validation"]
tools: ["scikit-learn"]
---

## What you will learn

- The three kinds of evidence about a clustering: internal, external and task-based.
- How the silhouette works, by hand on six points, and how to read a silhouette plot.
- Why inertia cannot pick $k$, and what to use instead.
- How to compare clusters with known labels using chance-corrected scores (ARI, AMI), and why purity and raw NMI mislead.
- How to test whether a clustering is stable enough to act on.

*Before you start:* k-means and inertia (05.01), DBSCAN (05.02) and Gaussian mixtures (05.03). The external-index section counts pairs ("$n$ choose 2").

This closes the module. Lessons 05.01 to 05.05 gave you many ways to find structure without labels, and every one of them returns an answer, even on data with no structure at all. This lesson is about deciding whether that answer is any good.

## Why this is hard

In supervised learning you hold out labelled data and count mistakes. Clustering has no answer key, and the same data can support several sensible clusterings (customers by spend, by region, at coarse or fine scale). So "is this clustering good?" splits into three questions.

~~~dg-layers Three levels of evidence, from cheapest to most convincing.
Internal | uses only the data and the cluster labels: inertia, silhouette, Davies-Bouldin
External | compares with reference classes you held back: ARI, NMI/AMI, homogeneity
*Task-based | do the clusters help the job: segment response rates, downstream model accuracy, expert review
~~~

The last level matters most in practice. The first two are quick tools for choosing among candidates (which $k$, which algorithm) and catching failures early.

## The idea in one picture

Take six numbers on a line: 1, 2, 4, 8, 9, 11. Your eye says there are two groups, a left one and a right one. Compare two ways to split them:

| Split | Left cluster | Right cluster | Mean silhouette
| --- | --- | --- | --- |
| A (natural) | 1, 2, 4 | 8, 9, 11 | 0.70
| B (one point moved) | 1, 2 | 4, 8, 9, 11 | 0.50

~~~dg-fig 05.06-toy.svg
Each point carries its silhouette value. In split B the point at 4 is on average closer to the other cluster than to its own, so its value turns negative (-0.53) and the mean falls from 0.70 to 0.50.
~~~

The **silhouette** of a point asks one question: is this point closer to its own cluster or to the nearest other cluster? It gives a number between -1 and 1. Near 1 means comfortably inside its cluster, near 0 means on a border, and negative means it probably belongs elsewhere. Averaging gives one score for the clustering, and split A wins. The figure also shows the most useful habit in this lesson: look at per-point values, not only the average. The single negative value names the wrong assignment.

## Step 1: inertia measures compactness only

**Inertia** (the within-cluster sum of squares from 05.01) adds up each point's squared distance to its own centroid:

$$\text{inertia} = \sum_{i=1}^{n} \lVert x_i - \mu_{c(i)} \rVert^2,$$

where $x_i$ is point $i$, $c(i)$ is the cluster it is assigned to, and $\mu_{c(i)}$ is that cluster's centroid. Smaller is tighter.

The catch: inertia only goes down as you add clusters, reaching 0 at $k = n$. So it cannot pick $k$; it only compares runs with the same $k$ (such as the best of ten k-means restarts). It also changes when you rescale features, so it cannot compare preprocessing choices either.

The **elbow method** plots inertia against $k$ and looks for a bend where extra clusters stop paying off. Below the bend is at $k = 4$, but real curves are often smooth slopes where two people pick two different $k$.

~~~dg-fig 05.06-choose-k.svg
Four blobs, k-means for k = 2 to 8. Inertia (left) only ever falls and shows a soft bend. The mean silhouette (right) rises to a clear peak at k = 4 and falls after it.
~~~

## Step 2: the silhouette balances tightness and separation

For point $i$ define two averages:

- $a(i)$: the mean distance from $i$ to the *other* points in its own cluster (how tight its home is).
- $b(i)$: the mean distance from $i$ to the points of the *nearest other* cluster (how far the closest neighbour cluster is). "Nearest" means the cluster with the smallest such mean.

Then

$$s(i) = \frac{b(i) - a(i)}{\max\big(a(i),\, b(i)\big)} \in [-1, 1].$$

The numerator is positive when the neighbour cluster is farther away than home; the denominator rescales it into $[-1, 1]$. A point alone in its cluster gets $s = 0$ by convention. The **silhouette score** is the mean of $s(i)$ over all points; it is undefined for $k = 1$ because there is no "other" cluster.

<div class="ex">
<div class="lab">Worked example</div>

**the silhouette by hand on six points**

Split A: clusters $\{1, 2, 4\}$ and $\{8, 9, 11\}$. Distances are absolute differences.

Point 1: $a = (1 + 3)/2 = 2$. $b = (7 + 8 + 10)/3 = 8.33$. $s = (8.33 - 2)/8.33 = 0.76$.

Point 2: $a = (1 + 2)/2 = 1.5$. $b = (6 + 7 + 9)/3 = 7.33$. $s = 5.83/7.33 = 0.80$.

Point 4: $a = (3 + 2)/2 = 2.5$. $b = (4 + 5 + 7)/3 = 5.33$. $s = 2.83/5.33 = 0.53$. The lowest on the left, because 4 is the left cluster's edge closest to the right group.

Point 8: $a = (1 + 3)/2 = 2$, $b = (7 + 6 + 4)/3 = 5.67$, $s = 0.65$. Point 9: $a = 1.5$, $b = (8 + 7 + 5)/3 = 6.67$, $s = 0.78$. Point 11: $a = (3 + 2)/2 = 2.5$, $b = (10 + 9 + 7)/3 = 8.67$, $s = 0.71$.

Mean: $(0.76 + 0.80 + 0.53 + 0.65 + 0.78 + 0.71)/6 = 4.23/6 = 0.70$.

Split B moves 4 to the right cluster. Now for point 4: $a = (4 + 5 + 7)/3 = 5.33$ (its new "home" is far away) and $b = (3 + 2)/2 = 2.5$. So $s = (2.5 - 5.33)/5.33 = -0.53$. A negative value is the silhouette saying "this point would fit better in the other cluster". The mean over all six drops to 0.50.

The same pattern scales up: a clustering whose mean silhouette rises from 0.31 at $k = 3$ to 0.48 at $k = 4$ and falls to 0.40 at $k = 5$ points to $k = 4$, provided the $k = 4$ silhouette plot has no cluster made mostly of negative values.

</div>

## Step 3: read the silhouette plot, not just the mean

An average can hide one bad cluster behind good ones. A **silhouette plot** sorts the $s(i)$ inside each cluster and draws them as horizontal bars, one block per cluster, with a dashed line at the mean.

~~~dg-fig 05.06-silplot.svg
Silhouette plots for k = 4 and k = 6 on the four-blob data. At k = 4 every block is thick and reaches well past the mean line. At k = 6, two real blobs were each cut in half: the halves are thin, short blocks that barely reach the (lower) mean.
~~~

What to look for:

- **How far right a block reaches**: a block mostly left of the mean line is a weak cluster.
- **Negative tails**: a few at borders are normal; a mostly negative block should not exist.
- **Thickness** (points per cluster): a very thin block often means one true group was split.

At $k = 4$ every cluster's mean silhouette is between 0.60 and 0.73 with almost no negatives; at $k = 6$ the mean drops to 0.50.

## Step 4: other internal indices

Two cheaper scores use centroids instead of all point pairs:

- **Davies-Bouldin** (lower is better): for each cluster, the worst ratio of (sum of two clusters' spreads) to (distance between their centroids), averaged. On the four-blob data it is 0.47 at $k = 4$, 0.67 at $k = 3$, 0.72 at $k = 5$.
- **Calinski-Harabasz** (higher is better): between-cluster over within-cluster spread, scaled by $(n - k)/(k - 1)$.

All internal indices encode a definition of "good cluster", usually *compact and roughly round*. That bias has a cost.

~~~dg-fig 05.06-moons.svg
Two half-moons. K-means draws a straight cut and gets the higher silhouette (0.48) while matching the true moons poorly (ARI 0.26). DBSCAN recovers both moons exactly (ARI 1.00) but scores only 0.33.
~~~

A long curved cluster has a large $a(i)$ (its two ends are far apart), so it looks "loose" and the correct answer is penalised. Internal indices are fair only among clusterings that share their shape assumption. When comparing DBSCAN or single linkage against k-means, trust a visual check (05.04) and the task more. Exclude DBSCAN noise points (label -1) before scoring and report the noise fraction separately.

## Step 5: external indices, when reference labels exist

Sometimes you hold labels not used for clustering (species, categories, a synthetic benchmark) and can measure agreement. A fair score must ignore label *names*: $(0, 0, 1, 1)$ and $(1, 1, 0, 0)$ are the same partition (ARI 1.0).

The cleanest idea counts **pairs of points**: for every pair, do the true classes put them together or apart, and does the clustering? The **Rand index** is the fraction of pairs on which the answers agree. With many groups most pairs are "apart" in both, so even random clusterings score high. The **adjusted Rand index (ARI)** subtracts what you would expect by chance:

$$\text{ARI} = \frac{\text{Index} - \text{Expected}}{\text{Max} - \text{Expected}},$$

where Index $= \sum_{ij} \binom{n_{ij}}{2}$ counts pairs together in both labellings ($n_{ij}$ is the number of points in class $i$ and cluster $j$), Expected is its average over random labellings with the same group sizes, and Max is its largest possible value. ARI is 1 for identical partitions, about 0 for random ones, and can be slightly negative.

<div class="ex">
<div class="lab">Worked example</div>

**Rand index and ARI by hand**

Six points. True classes: A A A B B B. Clusters: 1 1 2 2 2 2. Contingency counts: class A has 2 points in cluster 1 and 1 in cluster 2; class B has 0 in cluster 1 and 3 in cluster 2.

There are $\binom{6}{2} = 15$ pairs. Pairs together in both: $\binom{2}{2} + \binom{1}{2} + \binom{0}{2} + \binom{3}{2} = 1 + 0 + 0 + 3 = 4$.

Pairs together in the truth: $\binom{3}{2} + \binom{3}{2} = 6$. Pairs together in the clustering: $\binom{2}{2} + \binom{4}{2} = 1 + 6 = 7$.

Pairs apart in both: $15 - 6 - 7 + 4 = 6$. So the Rand index is $(4 + 6)/15 = 0.667$, which sounds respectable.

Expected "together in both" by chance: $6 \times 7 / 15 = 2.8$. Max: $(6 + 7)/2 = 6.5$.

ARI $= (4 - 2.8)/(6.5 - 2.8) = 1.2/3.7 = 0.32$. Only modest agreement once chance is removed. Purity, the fraction of points in their cluster's majority class, is $(2 + 3)/6 = 0.83$ and is even more flattering.

</div>

A second family uses information theory. **Mutual information** (MI) measures how much a point's cluster tells you about its class; **NMI** normalises it by the entropies, and **AMI** also subtracts the MI expected by chance. **Homogeneity**, **completeness** and their harmonic mean, **V-measure**, split agreement into two readable halves.

| Index | Random → perfect | Notes
| --- | --- | --- |
| Purity | depends on $k$ → 1 | Fraction in each cluster's majority class; 1 with one cluster per point, so never use alone
| Rand index | high → 1 | Fraction of point pairs on which the two labellings agree; inflated by the many "different" pairs
| Adjusted Rand index (ARI) | about 0 → 1 | Rand index corrected for chance; can be negative
| NMI | above 0 → 1 | Mutual information normalised by the entropies; biased upward with many clusters
| AMI | about 0 → 1 | NMI corrected for chance; prefer it with many clusters or small $n$
| Homogeneity / completeness / V-measure | 0 → 1 | Each cluster contains one class / each class lies in one cluster / their harmonic mean

The next figure shows why the chance correction matters. Every clustering in it is pure noise, labels drawn at random.

~~~dg-fig 05.06-chance.svg
Random clusterings scored against 10 true classes. As the number of random clusters grows, purity climbs to 0.52 and NMI to 0.35, rewarding nothing but fragmentation. ARI and AMI stay at 0 throughout.
~~~

So ARI and AMI are the defaults. A warning: if matching known classes were the goal, a classifier would be the right tool. Clusters that disagree with labels may find something real the labels miss (a sub-type, a collection batch). External scores are for benchmarking, not for declaring a clustering "correct".

## Evaluating clusters in code

All of these live in `sklearn.metrics`. The loop below scores k-means on the four-blob data for several $k$; `y_true` is only available because the data is synthetic.

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, silhouette_samples, davies_bouldin_score,
                             adjusted_rand_score, adjusted_mutual_info_score)

X, y_true = make_blobs(n_samples=600, centers=[[0, 0], [5, 0], [0, 5], [6.5, 6]],
                       cluster_std=[1.0, 1.0, 1.1, 0.9], random_state=4)

for k in range(2, 7):
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    print(k, silhouette_score(X, labels), davies_bouldin_score(X, labels),
          adjusted_rand_score(y_true, labels), adjusted_mutual_info_score(y_true, labels))
# k  silhouette  Davies-Bouldin  ARI    AMI
# 2  0.463       1.007           0.499  0.666
# 3  0.544       0.667           0.687  0.797
# 4  0.652       0.470           0.982  0.973   <- best on every score
# 5  0.569       0.718           0.899  0.921
# 6  0.501       0.895           0.806  0.868

labels = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(X)
s = silhouette_samples(X, labels)                 # one value per point
for c in range(4):
    print(c, s[labels == c].mean(), (s[labels == c] < 0).mean())
# per-cluster mean silhouette 0.63, 0.64, 0.73, 0.60; negative fraction <= 0.007
```

Internal and external scores agree here because the blobs are round. On large data pass `sample_size=5000` to `silhouette_score`, since it needs all pairwise distances.

## Step 6: stability, is the structure real?

A clustering worth acting on should come back when you repeat the analysis. Re-run with other seeds, on bootstrap resamples (draw $n$ points with replacement) or on two halves of the data, and compare labellings on shared points with ARI, which ignores label names.

On the four-blob data, two single-start k-means runs (`n_init=1`) with different seeds agree with ARI 0.996 at $k = 4$ but only 0.76 at $k = 6$: extra boundaries land somewhere new each time.

~~~dg-fig 05.06-stability.svg
Data with three real groups. Each dot is one bootstrap resample, scored by ARI against the full-data clustering. At k = 3 all 20 runs agree almost perfectly. At k = 2 the answer flips between merging one pair of groups or another, and at k = 4 to 7 boundaries wander.
~~~

Too few clusters can be unstable too: with $k = 2$ and three equal groups, which two merge is a coin toss. Too many clusters cut through continuous regions, and the cut moves with each resample.

If no $k$ is stable, the data may be a continuum rather than groups, and no internal index will tell you so.

## Step 7: evaluate against the purpose

Finally, judge the clusters by their job:

- **Segmentation**: do segments differ in behaviour that matters (conversion, churn) on *later* data, and can the team describe each one?
- **Features for a model**: does cluster membership improve the validation score (module 06)?
- **Exploration**: do domain experts recognise the groups from a few samples of each?

~~~dg-flow A practical order of checks for any clustering.
Fit candidates -(silhouette, Davies-Bouldin, plots)-> Shortlist -(seeds and resamples, ARI)-> Stable choices -(known labels, if any: ARI, AMI)-> Benchmarked -(task outcome)-> *Decision
~~~

## What goes wrong

| Mistake | Symptom | Fix
| --- | --- | --- |
| Choosing $k$ by lowest inertia | Always picks the largest $k$ tried | Use silhouette or Davies-Bouldin, elbow only as a hint
| Reporting only the mean silhouette | One bad cluster hidden by good ones | Draw the silhouette plot; check per-cluster means and negatives
| Silhouette to judge DBSCAN or single linkage | Correct curved clusters lose to wrong round ones | Visual check, stability and task metrics; exclude noise points
| Purity or raw NMI with many clusters | Scores rise as clusters fragment | Use ARI and AMI
| Exact silhouette on a million points | Out of memory: $O(n^2)$ distances | `sample_size=` a few thousand
| One run, one seed | Conclusions change on rerun | Repeat over seeds and resamples; compare with ARI
| Treating agreement with labels as success | "Good" clusters that only rediscover a known column | Ask what the clusters add; evaluate on the task

## Going deeper

**Cost.** The exact silhouette is $O(n^2)$; Davies-Bouldin and Calinski-Harabasz need only centroids, $O(nk)$. ARI and AMI work from the contingency table, though AMI's expected-MI term gets slow with many clusters.

**Where the chance corrections come from.** Both ARI and AMI use the **permutation model**: keep the class sizes and cluster sizes fixed, shuffle which points go where, and compute the expected score. Under that model the expected value of the pair-count index is $\big[\sum_i \binom{a_i}{2}\sum_j \binom{b_j}{2}\big]/\binom{n}{2}$, where $a_i$ and $b_j$ are the class and cluster sizes; that is the 2.8 in the hand example. AMI subtracts the expected MI under the same model. Vinh, Epps and Bailey (2010) showed raw NMI drifts upward as clusters grow relative to $n$, as in the chance figure.

**Distances.** The silhouette accepts any metric (`metric=`), such as cosine for embeddings; it is only as meaningful as that distance and the feature scaling behind it.

**The gap statistic** (Tibshirani, Walther and Hastie) compares $\log(\text{inertia}_k)$ with its expected value on uniform reference data. It formalises the elbow and, unlike the silhouette, can answer $k = 1$: "no clusters here".

**Model-based choice.** For Gaussian mixtures (05.03), use BIC on the likelihood to pick the number of components.

**No universal criterion.** Kleinberg's impossibility theorem (2002) shows no clustering function satisfies scale invariance, richness and consistency at once. Every method trades something off, so the task has the last word.

## Key terms

| Term | Meaning
| --- | --- |
| Inertia | Sum of squared distances from points to their centroids; falls whenever $k$ grows
| Silhouette $s(i)$ | $(b - a)/\max(a, b)$: own-cluster tightness against nearest-other-cluster distance, in $[-1, 1]$
| Silhouette plot | Sorted per-point silhouettes drawn per cluster, to spot weak or wrong clusters
| Davies-Bouldin | Average worst-case ratio of cluster spread to centroid separation; lower is better
| Rand index | Fraction of point pairs on which two labellings agree
| ARI | Rand index corrected for chance: about 0 for random, 1 for identical partitions
| NMI / AMI | Mutual information normalised to $[0, 1]$ / additionally corrected for chance
| Stability | How much the clustering changes across seeds or resamples, measured by ARI

<div class="take">
<div class="lab">Key takeaway</div>

use internal indices (silhouette and its plot, not raw inertia) to compare candidates that share a shape assumption, chance-corrected external indices (ARI, AMI) when reference labels exist, stability across resamples to check the structure is real, and task outcomes to decide whether it matters.

</div>
