---
id: "05.02"
course: "machine-learning"
module: "05"
slug: "hierarchical-and-density-based-clustering"
title: "Hierarchical and density-based clustering"
keyIdeas: ["Agglomerative clustering starts with every point as its own cluster and repeatedly merges the closest pair; the merge history is a dendrogram you can cut at any level, so $k$ is chosen afterwards.", "The linkage defines distance between clusters: single (closest pair, chains), complete (farthest pair, compact), average, and Ward (smallest increase in within-cluster variance, k-means-like).", "Agglomerative clustering needs the $O(n^2)$ distance matrix, which limits it to tens of thousands of points.", "DBSCAN defines clusters as dense regions: core points have at least `min_samples` neighbours within `eps`, clusters are connected groups of core points plus their border points, and the rest is labelled noise.", "DBSCAN finds arbitrary shapes and needs no $k$, but one `eps` cannot fit clusters of very different density; HDBSCAN removes that limit."]
question: "You cluster GPS points of taxi pick-ups in a city. Downtown has thousands of points per block; suburbs have a few per block. DBSCAN with `eps` tuned for downtown labels nearly all suburban points as noise; with `eps` tuned for the suburbs, downtown becomes one giant cluster. What is going on, and what would you use instead?"
answer: "DBSCAN uses one global density threshold: a point is core if it has `min_samples` neighbours within the same radius `eps` everywhere. Downtown and suburban clusters differ in density by orders of magnitude, so no single radius works. A small `eps` matches downtown's spacing, but suburban points then rarely have enough neighbours and become noise. A large `eps` captures the suburbs but connects all of downtown into one chain of core points. I would use HDBSCAN, which considers all density levels and keeps the clusters that are stable over a wide range of them, so dense downtown hotspots and sparse suburban ones can both appear, controlled mainly by `min_cluster_size`. OPTICS is an alternative. Because these are GPS coordinates, I would also use haversine distance, or project the points to metres, so that `eps` means the same distance everywhere."
sources: [{"title": "Ester, Kriegel, Sander, Xu (1996): A Density-Based Algorithm for Discovering Clusters (DBSCAN)", "url": "https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf", "note": "The original paper; definitions of core, border, density-reachability and the k-distance heuristic."}, {"title": "scikit-learn User Guide: Clustering", "url": "https://scikit-learn.org/stable/modules/clustering.html", "note": "Sections on hierarchical clustering (linkages, connectivity), DBSCAN, HDBSCAN and OPTICS."}, {"title": "scikit-learn example: Comparing different hierarchical linkage methods on toy datasets", "url": "https://scikit-learn.org/stable/auto_examples/cluster/plot_linkage_comparison.html", "note": "A picture of single, average, complete and Ward linkage on the same shapes."}, {"title": "HDBSCAN docs: How HDBSCAN Works", "url": "https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html", "note": "Step-by-step explanation of mutual reachability, the condensed tree and cluster stability."}]
practice: ["agglomerative-clustering", "dbscan"]
tags: ["hierarchical clustering", "linkage", "DBSCAN"]
tools: ["scikit-learn", "SciPy"]
---

## Two other definitions of a cluster

Lesson 05.01 defined a cluster as points near a centre. That fails for curved shapes, nested groups and noise. This lesson covers two alternatives: hierarchical clustering, which builds a whole tree of nested clusterings, and density-based clustering, which defines clusters as regions where points are packed closely.

## Agglomerative clustering

The bottom-up algorithm is simple:

1. Start with $n$ clusters, one per point, and the pairwise distance matrix.
2. Merge the two closest clusters.
3. Update the distances from the merged cluster to all others, using the linkage rule.
4. Repeat until one cluster remains.

The output is a dendrogram: a binary tree whose merge heights record the distance at which each merge happened. Cutting it at height $h$ gives the clustering in which no merge above $h$ has happened; cutting to get $k$ clusters undoes the last $k-1$ merges. Unlike k-means, you can inspect the structure at every scale and choose $k$ afterwards.

The cost is the catch. The distance matrix needs $O(n^2)$ memory, so 100,000 points means $10^{10}$ entries, which is 80 GB as float64. Naive merging is $O(n^3)$; efficient algorithms reach $O(n^2)$ for single linkage (via a minimum spanning tree) and $O(n^2 \log n)$ for others. In practice it is a tool for up to tens of thousands of points, or for clustering cluster centres produced by something cheaper.

## Linkage criteria

Everything interesting is in the linkage, the definition of distance between clusters $A$ and $B$:

| Linkage | Distance between clusters | Behaviour |
| --- | --- | --- |
| Single | $\min_{a \in A, b \in B} d(a, b)$ | Follows chains of nearby points; finds elongated shapes but one bridge of noise merges two clusters |
| Complete | $\max_{a \in A, b \in B} d(a, b)$ | Compact clusters of similar diameter; sensitive to outliers |
| Average (UPGMA) | mean of all pairwise distances | A compromise between single and complete; works with any metric |
| Ward | increase in total within-cluster sum of squares when merging | Similar-sized, spherical clusters, like k-means; Euclidean only |

Ward's criterion for merging clusters of sizes $n_A, n_B$ with means $\mu_A, \mu_B$ is $\frac{n_A n_B}{n_A + n_B}\lVert \mu_A - \mu_B \rVert^2$, which is exactly the inertia increase. So Ward is a greedy, hierarchical version of the k-means objective and the usual default; single linkage is the one to reach for when clusters are connected but irregular, and the data has little noise.

A connectivity constraint (for example, allowing merges only between $k$-nearest-neighbour graph neighbours) makes agglomerative clustering respect the geometry of the data, such as manifolds or neighbouring pixels, and is also much faster.

## DBSCAN

DBSCAN (Ester et al., 1996) needs two parameters: a radius `eps` and a count `min_samples`. It classifies points:

- **Core**: at least `min_samples` points (including itself, in scikit-learn's convention) lie within `eps`.
- **Border**: not core, but within `eps` of some core point.
- **Noise**: neither. Labelled −1 and assigned to no cluster.

Clusters are the connected components of core points, where two core points are connected if they are within `eps` of each other, plus the border points attached to them.

~~~dg-layers DBSCAN's three kinds of point. Only core points can extend a cluster.
*Core | ≥ min_samples neighbours within eps; links to other core points to grow the cluster
Border | within eps of a core point, too sparse to be core; joins that cluster, extends nothing
Noise | neither; label −1, belongs to no cluster
~~~

<div class="ex">
<div class="lab">Worked example</div>

**DBSCAN by hand**

One-dimensional points 1, 2, 3, 8, 20, 21, 22, 23 with eps = 1.5 and min_samples = 3.

Neighbourhoods: 2 sees {1, 2, 3}, so it is core; 1 and 3 see only two points each, so they are border points of 2's cluster. 21 sees {20, 21, 22} and 22 sees {21, 22, 23}: both core, linked, with 20 and 23 as borders. 8 sees only itself: noise.

Result: clusters {1, 2, 3} and {20, 21, 22, 23}, and one noise point. No k was given, and k-means with k = 2 would have dragged 8 into a cluster.

</div>

Strengths: arbitrary shapes, a built-in notion of noise, no $k$, and with a spatial index each neighbourhood query is fast, giving roughly $O(n \log n)$ in low dimensions (approaching $O(n^2)$ in high dimensions or with a large `eps`). Weaknesses:

- **One density for everything**: a single `eps` cannot fit a tight cluster and a diffuse cluster at once; the diffuse one fragments into noise or the tight ones merge.
- **Parameter choice**: set `min_samples` to about $2d$ or more, then plot each point's distance to its `min_samples`-th neighbour in sorted order and choose `eps` at the knee of that curve.
- **Border ambiguity**: a border point within `eps` of two clusters' cores goes to whichever is found first, so labels can depend on data order.
- **High dimensions**: distances concentrate and neighbourhoods become uninformative; reduce dimension first.

## HDBSCAN and OPTICS

HDBSCAN effectively runs DBSCAN at every `eps`, builds the hierarchy of clusters as density rises, and keeps the clusters that persist over the widest range of densities. It needs mainly `min_cluster_size`, handles clusters of varying density, and gives each point a membership strength. OPTICS computes a reachability ordering from which DBSCAN-style clusterings at any `eps` can be extracted. Both are in scikit-learn (`HDBSCAN`, `OPTICS`) and are usually better defaults than plain DBSCAN when densities differ.

<div class="take">
<div class="lab">Key takeaway</div>

agglomerative clustering gives a full merge tree whose shape is set by the linkage (Ward for compact clusters, single for chains) at $O(n^2)$ cost; DBSCAN finds dense regions of any shape and labels noise, but needs one density scale, which HDBSCAN removes.

</div>
