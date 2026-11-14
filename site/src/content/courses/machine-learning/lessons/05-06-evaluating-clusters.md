---
id: "05.06"
course: "machine-learning"
module: "05"
slug: "evaluating-clusters"
title: "Evaluating clusters"
keyIdeas: ["Cluster quality can be judged three ways: internal indices using only the data and labels (inertia, silhouette), external indices against known classes (ARI, NMI), and usefulness for the task the clusters serve.", "Inertia always falls with $k$ and only suits comparisons at fixed $k$. The silhouette $s = (b-a)/\\max(a,b)$ compares each point's mean distance to its own cluster ($a$) and to the nearest other cluster ($b$).", "Internal indices encode a definition of good clusters, usually compact and convex, so they penalise correct non-convex clusterings such as DBSCAN's.", "External indices must be invariant to label permutation and corrected for chance: purity is trivially 1 with one cluster per point, while ARI and AMI are about 0 for random labels.", "Check stability: re-cluster bootstrap samples or other seeds and compare runs with ARI. Structure that changes between runs is not structure."]
question: "Two clusterings of the same 10,000 documents, compared with 20 known topic labels: A is k-means with k=20 (purity 0.62, NMI 0.48, ARI 0.31); B is a k-means run with k=500 (purity 0.81, NMI 0.58, ARI 0.09). A colleague prefers B because two of the three numbers are higher. Who is right?"
answer: "A is the better match to the topics. Purity and NMI both rise mechanically with the number of clusters. With 500 clusters of about 20 documents each, most clusters are dominated by one topic, so purity is high, and NMI's normalisation is known to be biased upward when there are many clusters. Neither is corrected for chance. ARI is chance-corrected and counts how many document pairs the clustering groups the same way as the topic labels do. Splitting each topic into about 25 fragments breaks most same-topic pairs, so ARI drops to 0.09. To compare the two fairly I would report AMI, which corrects NMI for chance, alongside ARI, and both should favour A. If fine-grained clusters are actually wanted, for example as features or for retrieval, that should be evaluated on the task itself, not by agreement with 20 topics."
sources: [{"title": "scikit-learn User Guide: Clustering performance evaluation", "url": "https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation", "note": "Definitions, pros and cons of ARI, AMI/NMI, homogeneity, silhouette, Calinski-Harabasz and Davies-Bouldin."}, {"title": "scikit-learn example: Selecting the number of clusters with silhouette analysis", "url": "https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html", "note": "Silhouette plots for several k, showing how to read weak clusters."}, {"title": "Vinh, Epps, Bailey (2010): Information Theoretic Measures for Clusterings Comparison", "url": "https://jmlr.org/papers/v11/vinh10a.html", "note": "Why NMI needs a chance adjustment, and the case for AMI."}]
practice: ["k-means-clustering", "agglomerative-clustering", "gaussian-mixture-models-em", "dbscan"]
tags: ["silhouette", "adjusted Rand index", "cluster validation"]
tools: ["scikit-learn"]
---

## Why this is hard

In supervised learning, a held-out label tells you whether a prediction is right. Clustering has no such label, and the same data can admit several reasonable clusterings at different scales or along different features. So "evaluating clusters" means one of three different things:

~~~dg-layers Three levels of evidence, from cheapest to most convincing.
Internal | uses only the data and the cluster labels: inertia, silhouette, Davies-Bouldin
External | compares with reference classes you held back: ARI, NMI/AMI, homogeneity
*Task-based | do the clusters help the job: segment response rates, downstream model accuracy, expert review
~~~

The last level is the one that matters in practice. The first two are tools for choosing among candidates and catching failures quickly.

## Internal indices

**Inertia** (within-cluster sum of squares, lesson 05.01) measures compactness only. It decreases monotonically with $k$, and is not comparable across scalings or algorithms, so use it to compare k-means runs at the same $k$, or to draw an elbow curve.

**Silhouette** balances compactness against separation. For point $i$, let $a(i)$ be its mean distance to the other points in its cluster and $b(i)$ its mean distance to the points in the nearest other cluster. Then

$$s(i) = \frac{b(i) - a(i)}{\max\big(a(i), b(i)\big)} \in [-1, 1]$$

Near 1 means well inside its cluster; near 0 means on a boundary; negative means it is on average closer to another cluster, which suggests misassignment. The mean over all points is the silhouette score, and plotting the sorted $s(i)$ per cluster (a silhouette plot) shows which clusters are weak.

<div class="ex">
<div class="lab">Worked example</div>

**silhouette for two points**

Point P: mean distance 1.0 to its own cluster, 3.0 to the nearest other cluster. s = (3.0 − 1.0) / 3.0 = 0.67, well placed.

Point Q: mean distance 2.0 to its own cluster, 1.5 to the nearest other. s = (1.5 − 2.0) / 2.0 = −0.25. Q would fit better in the other cluster.

A clustering whose mean silhouette rises from 0.31 at k = 3 to 0.48 at k = 4 and falls to 0.40 at k = 5 points to k = 4, provided the k = 4 silhouette plot has no cluster made mostly of negative values.

</div>

Cost and caveats: the exact silhouette needs all pairwise distances, $O(n^2)$, so sample a few thousand points for large data (`sample_size` in scikit-learn). It is undefined for $k = 1$, and it rewards convex, well-separated clusters: two interleaved half-moons correctly found by DBSCAN can score lower than a wrong k-means split. DBSCAN noise points should be excluded or reported separately. Davies-Bouldin (lower is better) and Calinski-Harabasz (higher is better) are cheaper alternatives built on centroids, with the same convexity bias.

## External indices

When reference labels exist, for example to benchmark an algorithm or check that clusters recover known categories, compare the clustering with them. Any index must ignore label names, since cluster 3 might correspond to class A.

| Index | Range (random → perfect) | Notes |
| --- | --- | --- |
| Purity | depends on $k$ → 1 | Fraction in each cluster's majority class; 1 with one cluster per point, so never use alone |
| Rand index | high → 1 | Fraction of point pairs on which the two labellings agree; inflated by the many "different" pairs |
| Adjusted Rand index (ARI) | about 0 → 1 | Rand index corrected for chance; can be negative |
| NMI | above 0 → 1 | Mutual information normalised by the entropies; biased upward with many clusters |
| AMI | about 0 → 1 | NMI corrected for chance; prefer it with many clusters or small $n$ |
| Homogeneity / completeness / V-measure | 0 → 1 | Each cluster contains one class / each class lies in one cluster / their harmonic mean |

ARI and AMI are the usual defaults. They are chance-corrected, so a random clustering scores near 0 no matter how many clusters it has, which is exactly what purity and raw NMI fail to do.

```python
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                             adjusted_mutual_info_score, davies_bouldin_score)
sil = silhouette_score(X, labels, sample_size=5000, random_state=0)
ari = adjusted_rand_score(y_true, labels)             # symmetric, permutation-invariant
ami = adjusted_mutual_info_score(y_true, labels)
```

Do not let external validation turn into "the clustering is good because it matches the classes". If the classes were what you wanted, a classifier would be the right tool. Clusters that disagree with known labels may reveal something new, or may be noise; task-based checks decide which.

## Stability and the task

A clustering worth acting on should be reproducible. Re-run it with different seeds, on bootstrap resamples, or on two halves of the data, and compare the labellings on shared points with ARI. If the ARI between runs is low, the algorithm is carving arbitrary boundaries through a continuum, and no internal index will tell you that.

Finally, evaluate against the purpose:

- **Segmentation**: do segments differ in behaviour that matters (conversion, churn) on later data, and can the team describe each one?
- **Features for a model**: does adding cluster membership or distances improve validation score?
- **Data exploration**: do domain experts recognise the groups when shown samples from each?

<div class="take">
<div class="lab">Key takeaway</div>

use internal indices (silhouette, not raw inertia) to compare candidates, chance-corrected external indices (ARI, AMI) when reference labels exist, stability across resamples to check the structure is real, and task outcomes to decide whether it matters.

</div>
