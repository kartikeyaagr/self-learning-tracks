---
id: "05.02"
course: "machine-learning"
module: "05"
slug: "hierarchical-and-density-based-clustering"
title: "Hierarchical and density-based clustering"
keyIdeas: ["Agglomerative clustering starts with every point as its own cluster and repeatedly merges the two closest clusters. The merge history is a **dendrogram** you can cut at any height, so the number of clusters is chosen afterwards.", "The **linkage** defines distance between clusters: single (closest pair, follows chains), complete (farthest pair, compact), average (mean of all pairs), and Ward (smallest increase in within-cluster sum of squares, k-means-like and the usual default).", "Agglomerative clustering needs the $O(n^2)$ table of pairwise distances, which limits it to tens of thousands of points unless you sample, add a connectivity graph or cluster k-means centres.", "DBSCAN defines clusters as crowded regions: **core points** have at least `min_samples` points within radius `eps`, clusters are chains of core points plus their **border points**, and everything else is labelled noise (−1). Choose `eps` at the knee of the sorted k-distance plot.", "DBSCAN finds any shape and needs no $k$, but one `eps` cannot fit clusters of very different density; HDBSCAN (and OPTICS) consider all densities and keep the most stable clusters."]
question: "You cluster GPS points of taxi pick-ups in a city. Downtown has thousands of points per block; suburbs have a few per block. DBSCAN with `eps` tuned for downtown labels nearly all suburban points as noise; with `eps` tuned for the suburbs, downtown becomes one giant cluster. What is going on, and what would you use instead?"
answer: "DBSCAN uses one global density threshold: a point is core if it has `min_samples` neighbours within the same radius `eps` everywhere in the city. Downtown and suburban clusters differ in density by orders of magnitude, so no single radius works. A small `eps` matches downtown's spacing, but suburban points then rarely have enough neighbours inside that small circle, fail the core test and become noise. A large `eps` captures the suburbs, but downtown then has so many core points within reach of each other that they link into one long chain, and separate hotspots fuse into a single cluster. A k-distance plot would show this directly: instead of one clear knee there would be two or more, one per density level. I would use HDBSCAN, which builds the cluster hierarchy over all density levels (via core distances and mutual reachability) and keeps the clusters that stay stable over the widest range of them, so dense downtown hotspots and sparse suburban ones can both appear. It is controlled mainly by `min_cluster_size` (the smallest group worth calling a cluster, say 30 pick-ups) and gives a membership strength per point. OPTICS is an alternative that exposes the same structure as a reachability plot. Because these are latitude and longitude, I would also use the haversine distance (in radians) or project the points to metres first, so that the radius means the same physical distance everywhere, and I would check the noise fraction and cluster sizes per district before trusting the result."
sources: [{"title": "Ester, Kriegel, Sander, Xu (1996): A Density-Based Algorithm for Discovering Clusters (DBSCAN)", "url": "https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf", "note": "The original paper; definitions of core, border, density-reachability and the k-distance heuristic."}, {"title": "scikit-learn User Guide: Clustering", "url": "https://scikit-learn.org/stable/modules/clustering.html", "note": "Sections on hierarchical clustering (linkages, connectivity), DBSCAN, HDBSCAN and OPTICS."}, {"title": "scikit-learn example: Comparing different hierarchical linkage methods on toy datasets", "url": "https://scikit-learn.org/stable/auto_examples/cluster/plot_linkage_comparison.html", "note": "A picture of single, average, complete and Ward linkage on the same shapes."}, {"title": "HDBSCAN docs: How HDBSCAN Works", "url": "https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html", "note": "Step-by-step explanation of mutual reachability, the condensed tree and cluster stability."}]
practice: ["agglomerative-clustering", "dbscan"]
tags: ["hierarchical clustering", "linkage", "DBSCAN"]
tools: ["scikit-learn", "SciPy"]
---

## What you will learn

- Why "points near a centre" (k-means) is only one possible definition of a cluster, and what the two main alternatives are.
- How agglomerative clustering builds a **dendrogram** by merging the closest groups, and how to cut it to get any number of clusters.
- What a **linkage** is, and how single, complete, average and Ward linkage change the shapes you find.
- How DBSCAN labels points as core, border or noise, and how to choose its two settings, `eps` and `min_samples`.
- Why DBSCAN struggles when clusters have different densities, and how HDBSCAN and OPTICS fix that.

*Before you start:* you need Euclidean distance and the k-means lesson (05.01), especially its idea of within-cluster sum of squares (inertia). Nearest-neighbour searches from the k-NN lesson help but are not required.

Lesson 05.01 defined a cluster as points close to a shared centre. That quietly assumes round blobs of similar size, and forces every point into some cluster. Real data breaks this: groups nest inside bigger groups, GPS tracks curve, and sensor logs contain junk readings that belong nowhere. Two alternatives: **hierarchical clustering** builds a whole family of nested clusterings at once, and **density-based clustering** calls a cluster any crowded region surrounded by emptier space.

## The idea in one picture

Start with the hierarchical idea on data small enough to do in your head. Five shops sit along one street:

| Shop | A | B | C | D | E
| --- | --- | --- | --- | --- | --- |
| Position (km) | 1 | 2 | 4.5 | 9 | 10.5

The gaps between neighbours are 1, 2.5, 4.5 and 1.5 km. Agglomerative clustering repeats one move: join the two closest groups. First A and B (1 km apart), then D and E (1.5 km), then C joins A and B, and finally the two remaining groups join. Write down the distance at each join and you get a tree, the **dendrogram**.

~~~dg-fig 05.02-dendro.svg
Left: the five shops and the gaps between them. Middle and right: the merge trees under two linkage rules. The order of merges is the same, but complete linkage records the later merges higher. A horizontal cut through either tree gives two clusters, {A, B, C} and {D, E}.
~~~

Read a dendrogram from the bottom up. Each leaf is a data point; each horizontal bar is a merge, drawn at the distance where it happened. A long vertical stretch with no bars means "nothing else is near this group for a while", which is what a real cluster looks like.

To get flat clusters, **cut** the tree with a horizontal line: each vertical line it crosses becomes one cluster. Cutting the middle tree at 3.5 km gives {A, B, C} and {D, E}; cutting at 2 km gives {A, B}, {C}, {D, E}. You choose the number of clusters $k$ afterwards, by looking at the tree.

The density idea needs a different picture. Two interleaved half-moons with some scattered junk points defeat k-means, because no pair of centres can describe a curved shape:

~~~dg-fig 05.02-moons.svg
Left: k-means with k = 2 draws a straight boundary and cuts each moon in half. Right: DBSCAN follows each moon because it only asks "is this region crowded?", and it marks the scattered points as noise (grey crosses).
~~~

## Step 1: the agglomerative algorithm

"Agglomerative" means "gathering together": the algorithm works bottom-up.

1. Start with $n$ clusters, one per point, and compute the distance between every pair of points.
2. Find the two closest clusters and merge them into one.
3. Work out the distance from the new, merged cluster to every other cluster, using a **linkage** rule (Step 2).
4. Repeat steps 2 and 3 until a single cluster remains. That takes exactly $n - 1$ merges.

The record of all $n - 1$ merges and their heights is the dendrogram. Cutting it to get $k$ clusters is the same as undoing the last $k - 1$ merges. Cutting at height $h$ gives the clustering in which no merge above $h$ has happened.

## Step 2: the linkage decides everything

Two single points are $d(a, b)$ apart. For groups we need a rule for "distance between two groups": the **linkage**. For clusters $A$ and $B$:

- **Single linkage**: the distance between their *closest* pair, $\min_{a \in A,\, b \in B} d(a, b)$.
- **Complete linkage**: the distance between their *farthest* pair, $\max_{a \in A,\, b \in B} d(a, b)$.
- **Average linkage** (also called UPGMA): the mean of all $|A| \cdot |B|$ pairwise distances, where $|A|$ is the number of points in $A$.
- **Ward linkage**: how much the total within-cluster sum of squares (the k-means inertia) would grow if $A$ and $B$ were merged.

For Ward, with $n_A, n_B$ the cluster sizes and $\mu_A, \mu_B$ their means, the growth in inertia is

$$\Delta(A, B) = \frac{n_A\, n_B}{n_A + n_B}\, \lVert \mu_A - \mu_B \rVert^2.$$

So Ward is a greedy version of the k-means objective: each step makes the merge that hurts compactness least. Two single points 1 km apart cost $\frac{1 \cdot 1}{2} \cdot 1^2 = 0.5$.

<div class="ex">
<div class="lab">Worked example</div>

**single versus complete linkage on the five shops**

Positions A = 1, B = 2, C = 4.5, D = 9, E = 10.5. Closest pair overall: A–B at 1, so merge {A, B} at height 1 (both linkages agree, since the clusters are single points).

Next candidates: {A, B}–C, C–D = 4.5, D–E = 1.5. Under single linkage {A, B}–C = min(|1 − 4.5|, |2 − 4.5|) = min(3.5, 2.5) = 2.5. Under complete linkage it is max(3.5, 2.5) = 3.5. Either way D–E at 1.5 is smallest, so merge {D, E} at 1.5.

Third merge, single linkage: {A, B}–C = 2.5, C–{D, E} = min(4.5, 6) = 4.5, {A, B}–{D, E} = min over pairs = 9 − 2 = 7. Smallest is 2.5: C joins {A, B} at height 2.5.

Third merge, complete linkage: {A, B}–C = 3.5, C–{D, E} = max(4.5, 6) = 6, {A, B}–{D, E} = 10.5 − 1 = 9.5. Smallest is 3.5: C joins {A, B} at height 3.5.

Last merge: single linkage uses the closest pair, C to D = 9 − 4.5 = 4.5. Complete linkage uses the farthest pair, A to E = 10.5 − 1 = 9.5.

Heights: single (1, 1.5, 2.5, 4.5), complete (1, 1.5, 3.5, 9.5). Same tree shape, different heights. Average linkage gives (1, 1.5, 3, 7.25); 7.25 is the mean of the six cross-distances 8, 9.5, 7, 8.5, 4.5, 6 (sum 43.5).

</div>

On a tiny, well-separated example all linkages agree on the tree. On real shapes they disagree sharply:

~~~dg-fig 05.02-linkage.svg
The same two datasets clustered with four linkages. Top: only single linkage follows the curved moons. Bottom: a thin bridge of points joins two blobs, and single linkage walks straight across it, merging them; complete, average and Ward keep the blobs apart.
~~~

| Linkage | Distance between clusters | Behaviour
| --- | --- | --- |
| Single | closest pair | Follows chains of nearby points; finds elongated shapes, but one bridge of noise merges two clusters ("chaining")
| Complete | farthest pair | Compact clusters of similar diameter; sensitive to outliers, which make a cluster look large
| Average | mean of all pairs | A compromise between single and complete; works with any distance
| Ward | growth in within-cluster sum of squares | Similar-sized, roughly round clusters, like k-means; Euclidean distance only

Ward is scikit-learn's default. Use single linkage for connected, irregular clusters with little noise, and average linkage when you need a non-Euclidean distance (cosine for text, or a custom distance matrix).

## Step 3: what it costs

The algorithm needs all pairwise distances: $n^2$ entries. For 100,000 points that is $10^{10}$ numbers, 80 GB as 64-bit floats. A naive version takes $O(n^3)$ time, since each of the $n - 1$ merges searches the table. Good implementations reach $O(n^2)$ for single linkage (via a minimum spanning tree) and about $O(n^2 \log n)$ for the others.

So it is a tool for up to tens of thousands of points. For more, cluster a sample, or run k-means with a few hundred centres and build the hierarchy on those. A **connectivity constraint** (merging only neighbours in a $k$-nearest-neighbour graph, or neighbouring pixels in an image) makes it respect the data's geometry and skip most of the table.

## Step 4: density-based clustering with DBSCAN

DBSCAN (Ester et al., 1996; the name is short for density-based spatial clustering of applications with noise) needs two settings: a radius `eps` and a count `min_samples`. For every point it draws a circle of radius `eps` and counts how many points fall inside, including the point itself (scikit-learn's convention). Then it sorts points into three kinds:

- A **core point** has at least `min_samples` points within `eps`. Core points sit in the crowded interior of a cluster.
- A **border point** is not core itself, but lies within `eps` of some core point. It is on the edge of a cluster.
- A **noise point** is neither. It gets the label −1 and belongs to no cluster.

~~~dg-fig 05.02-core.svg
Ten points with eps = 0.8 and min_samples = 4. The four packed points are core (solid circles); the two points at the group's edge are border points; the four isolated points are noise. Note the point at (0.6, 0.8): it is only 1.0 from the nearest core point, just outside the radius.
~~~

Clusters are built by linking: core points within `eps` of each other, directly or through a chain of core points, share a cluster. Each border point joins the cluster of a nearby core point, but never extends it further.

~~~dg-layers DBSCAN's three kinds of point. Only core points can extend a cluster.
*Core | ≥ min_samples neighbours within eps; links to other core points to grow the cluster
Border | within eps of a core point, too sparse to be core; joins that cluster, extends nothing
Noise | neither; label −1, belongs to no cluster
~~~

<div class="ex">
<div class="lab">Worked example</div>

**DBSCAN by hand**

One-dimensional points 1, 2, 3, 8, 20, 21, 22, 23 with eps = 1.5 and min_samples = 3.

Point 2 sees {1, 2, 3} (each within 1), so it has 3 points and is core. Point 1 sees only {1, 2}, because 3 is 2 away, which is more than 1.5; same for point 3. They are border points of 2's cluster.

Point 21 sees {20, 21, 22} and point 22 sees {21, 22, 23}: both core, and they are 1 apart, so they link into one cluster. Points 20 and 23 see only two points each: border points of that cluster.

Point 8 sees only itself: noise.

Result: clusters {1, 2, 3} and {20, 21, 22, 23} plus one noise point, with no $k$ given. k-means with k = 2 would have forced 8 into a cluster.

</div>

Because a cluster is just a connected crowded region, it can have any shape, noise is built in, and you never give $k$. With a spatial index (k-d tree or ball tree) the run is roughly $O(n \log n)$ in low dimensions, drifting towards $O(n^2)$ in high dimensions or when `eps` is so large that every point sees most others.

## Step 5: choosing eps and min_samples

Pick `min_samples` first: about $2d$ for $d$ features, at least 3 or 4, and higher for noisy data (a stricter core test ignores small accidental clumps).

Then pick `eps` with a **k-distance plot**. For every point, compute the distance to its `min_samples`-th nearest neighbour (counting itself), sort those distances, and plot them.

~~~dg-fig 05.02-kdist.svg
The k-distance plot for the moons data with min_samples = 5. Points inside the moons have small, similar distances (the long flat part); noise points have large ones (the sharp rise). A radius near the knee separates the two.
~~~

Choose `eps` near the **knee**, where flat turns into steep. Here the knee is around 0.15, and that value gives the two clean moons in the earlier figure, with 13 of 325 points labelled noise. One giant cluster means `eps` is too large; mostly noise or dozens of tiny clusters means it is too small (or `min_samples` too large).

## Hierarchical and density clustering in code

SciPy keeps the whole tree, which is what you want for exploring; scikit-learn gives you labels directly.

```python
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import AgglomerativeClustering, DBSCAN, HDBSCAN
from sklearn.datasets import make_moons

shops = np.array([[1], [2], [4.5], [9], [10.5]])          # positions in km, shape (n, 1)
Z = linkage(shops, method='complete')                       # SciPy keeps the whole merge tree
print(Z[:, 2])                                              # merge heights [1.  1.5 3.5 9.5]
print(fcluster(Z, t=2, criterion='maxclust'))               # cut into 2 clusters: [2 2 2 1 1]
print(fcluster(Z, t=5, criterion='distance'))               # cut at height 5:     [2 2 2 1 1]

agg = AgglomerativeClustering(n_clusters=2, linkage='single').fit(shops)
print(agg.labels_)                                          # [0 0 0 1 1]  (label numbers are arbitrary)

x = np.array([1, 2, 3, 8, 20, 21, 22, 23.]).reshape(-1, 1)
db = DBSCAN(eps=1.5, min_samples=3).fit(x)
print(db.labels_)                                           # [ 0  0  0 -1  1  1  1  1]
print(db.core_sample_indices_)                              # [1 5 6]  -> the points 2, 21, 22

X, _ = make_moons(300, noise=0.06, random_state=1)
X = np.vstack([X, np.random.default_rng(0).uniform([-1.3, -0.8], [2.3, 1.3], (25, 2))])
lab = DBSCAN(eps=0.15, min_samples=5).fit_predict(X)
print(np.unique(lab, return_counts=True))                   # labels [-1 0 1], counts [13 153 159]
hd = HDBSCAN(min_cluster_size=20).fit(X)                    # no eps needed
print(np.unique(hd.labels_, return_counts=True))            # labels [-1 0 1], counts [42 139 144]
```

Also useful: `dendrogram(Z)` draws the tree; `AgglomerativeClustering(n_clusters=None, distance_threshold=5)` cuts at a height; HDBSCAN's `probabilities_` gives each point a 0 to 1 membership strength. Standardise features first (module 07): everything here runs on distances.

## Reading the result

- **Look for long vertical gaps in the dendrogram.** A good cut crosses the tree where the branches are tallest, because those clusters stay separate over a wide range of distances. If merge heights rise smoothly with no big jump, the data may have no strong cluster structure at all.
- **Count the noise.** With DBSCAN, report how many points are labelled −1. A few percent is normal; half the data means `eps` is too small for at least part of it.
- **Check sizes.** One cluster holding 95% of the points usually means chaining or an `eps` that is too large.
- **Plot it**, in 2-D or via a projection (05.04), which can hide or invent separations.
- **Score with care.** Silhouette (05.06) favours round clusters, so it can rank a correct DBSCAN result on moons below a wrong k-means one. Exclude noise before scoring.

## What goes wrong

The biggest DBSCAN failure is clusters of different density. One `eps` is one density threshold, applied everywhere.

~~~dg-fig 05.02-density.svg
Two tight clusters and one diffuse cluster. Left: a small eps finds the tight pair but turns the diffuse cluster into noise (grey). Middle: a large eps captures the diffuse cluster but fuses the two tight ones. Right: HDBSCAN, with no eps at all, finds all three.
~~~

| Symptom | Likely cause | Fix
| --- | --- | --- |
| One huge cluster swallows everything (single linkage) | chaining through a few bridge points | use Ward or average linkage, or remove noise first
| Clusters change completely after scaling one feature | distances dominated by the feature with the largest units | standardise features before clustering
| Most points are DBSCAN noise | `eps` too small, or `min_samples` too large for the sparse parts | read the k-distance plot; try HDBSCAN
| Dense clusters merged while sparse ones are noise | clusters of very different density | HDBSCAN or OPTICS
| A border point flips cluster between runs | it is within `eps` of two clusters' cores; it goes to whichever is reached first | accept it, or treat border points as uncertain
| DBSCAN finds nothing useful in 100+ dimensions | distances concentrate: everything is about equally far | reduce dimension first (PCA, embeddings)
| Out of memory on 200,000 points | agglomerative clustering stores $n^2$ distances | sample, use a connectivity graph, or cluster k-means centres

## Going deeper

**One update rule for all linkages.** After merging $i$ and $j$, the distance from any other cluster $k$ to the new cluster can be computed from old distances alone, using the Lance-Williams formula

$$d(k, i \cup j) = \alpha_i\, d(k, i) + \alpha_j\, d(k, j) + \beta\, d(i, j) + \gamma\, \lvert d(k, i) - d(k, j) \rvert.$$

With $\alpha_i = \alpha_j = \tfrac12$, $\beta = 0$ and $\gamma = -\tfrac12$ this is exactly $\min(d(k,i), d(k,j))$: single linkage. Setting $\gamma = +\tfrac12$ gives the maximum: complete linkage. Average and Ward linkage use size-weighted $\alpha$ values. This is why the whole family fits in one implementation: no need to revisit the raw points.

**Single linkage is a minimum spanning tree.** Build the minimum spanning tree of the points (the cheapest set of edges connecting all of them). Deleting its $k - 1$ longest edges leaves exactly the $k$ single-linkage clusters. This gives the $O(n^2)$ algorithm and explains chaining: one short edge is enough to join two groups.

**Ward heights in SciPy** are $\sqrt{2\Delta}$, so two single points merge at their ordinary distance. On the shops C joins {A, B} at $\sqrt{2 \times 6} \approx 3.46$, where $6 = \frac{2 \cdot 1}{3} \cdot 3^2$.

**DBSCAN, formally.** $q$ is *density-reachable* from a core point $p$ if a chain of core points, each within `eps` of the next, leads from $p$ to $q$. A cluster is a maximal set of points all reachable from a common core point. Core points get the same clusters in any data order; only border points can depend on it.

**HDBSCAN.** It replaces `eps` with a search over all densities. Each point gets a **core distance**: the distance to its `min_samples`-th neighbour, small in crowded areas and large in sparse ones. The **mutual reachability distance** between $a$ and $b$ is $\max(\text{core}(a), \text{core}(b), d(a, b))$, which pushes sparse points away from everything while leaving dense regions unchanged. HDBSCAN builds single linkage on this distance, discards splits that create pieces smaller than `min_cluster_size`, and keeps the clusters that persist over the widest range of densities (their **stability**). In effect it runs DBSCAN at every `eps` and keeps, for each region, the scale that fits best.

**OPTICS** orders points so each is visited next to its dense neighbours and records a *reachability distance*; plotted in order, clusters appear as valleys, and DBSCAN at any `eps` is a horizontal line through the plot. Both are in scikit-learn and are better defaults than DBSCAN when densities vary.

## Key terms

| Term | Meaning
| --- | --- |
| Agglomerative clustering | Bottom-up clustering that repeatedly merges the two closest clusters
| Dendrogram | The merge tree; bar heights are the distances at which merges happened
| Linkage | The rule for the distance between two clusters (single, complete, average, Ward)
| Chaining | Single linkage joining two groups through a thin line of points
| Core point | A point with at least `min_samples` points within `eps`
| Border point | A non-core point within `eps` of a core point; joins that cluster
| Noise point | Neither core nor border; DBSCAN label −1
| k-distance plot | Sorted distances to the `min_samples`-th neighbour, used to choose `eps`
| HDBSCAN | Density clustering over all `eps` values at once, keeping the most stable clusters

<div class="take">
<div class="lab">Key takeaway</div>

agglomerative clustering builds a full merge tree whose shape is set by the linkage (Ward for compact clusters, single for chains) and costs $O(n^2)$ memory; DBSCAN finds crowded regions of any shape and labels the rest as noise, but uses one density scale everywhere, a limit that HDBSCAN removes.

</div>
