---
id: "16.03"
course: "ml-maths-course"
module: "16"
slug: "spectral-clustering"
title: "Spectral clustering"
keyIdeas: ["Spectral clustering builds a similarity graph over data points, embeds each point using the smallest eigenvectors of a Laplacian, then runs k-means in that embedding.", "It solves a relaxed version of a graph cut problem: separate the points into groups with little similarity between groups and enough points in each group.", "It can find non-convex clusters, such as rings and interleaved moons, that k-means on raw features cannot.", "The choices that matter most are the similarity graph (neighbours and kernel width) and the number of clusters, for which the eigengap is a useful guide."]
question: "You cluster 2,000 points shaped as two interleaved half-moons. k-means with $k = 2$ splits each moon in half. (a) Explain why. (b) Describe a spectral clustering pipeline that separates the moons, including how you build the graph. (c) The first eigenvalues of the normalised Laplacian are 0, 0.002, 0.31, 0.35, 0.40. How many clusters does the eigengap suggest?"
answer: "(a) k-means assigns each point to the nearest centroid, which produces convex regions separated by a straight boundary; no straight line separates two interleaved moons. (b) Build a symmetric $k$-nearest-neighbour graph (for example 10 neighbours) with Gaussian weights $w_{ij} = \\exp(-\\|x_i - x_j\\|^2 / 2\\sigma^2)$, where $\\sigma$ is around the typical neighbour distance. Points within one moon are linked in a chain; few edges cross between moons. Compute the two smallest eigenvectors of the symmetric normalised Laplacian, normalise each row of the resulting $2000 \\times 2$ matrix to unit length, and run k-means with $k = 2$ on the rows. (c) Two: the first two eigenvalues are near zero and there is a large jump from 0.002 to 0.31 before the rest, which are similar to each other."
sources: [{"title": "von Luxburg, A Tutorial on Spectral Clustering (2007)", "url": "https://arxiv.org/abs/0711.0189", "note": "The standard reference, including graph construction and the cut view."}, {"title": "Wikipedia, Spectral clustering", "url": "https://en.wikipedia.org/wiki/Spectral_clustering", "note": "Algorithm summary and variants."}, {"title": "Hamilton, Graph Representation Learning (free book)", "url": "https://www.cs.mcgill.ca/~wlh/grl_book/", "note": "Spectral methods as a foundation for graph embeddings."}]
practice: ["rbf-kernel-matrix", "k-means-clustering", "eigendecomposition", "k-means-initialization"]
tags: ["spectral clustering", "similarity graphs", "normalised cut", "eigengap"]
tools: ["NumPy", "scikit-learn"]
---

## Why another clustering method

k-means finds compact, roughly spherical clusters. Many real groupings are not like that: communities in a network, objects in an image, manifold-shaped data. Spectral clustering instead asks for groups that are internally connected and weakly connected to each other, where "connected" is defined by a similarity graph. It turns the problem into linear algebra on the graph Laplacian from lesson 16.02.

## The algorithm

1. Build a similarity graph over the $n$ points.
2. Form a Laplacian, usually the symmetric normalised one, $L_{\text{sym}} = I - D^{-1/2} W D^{-1/2}$.
3. Compute the $k$ eigenvectors with the smallest eigenvalues and stack them as columns of an $n \times k$ matrix $U$.
4. Normalise each row of $U$ to unit length (this step, from Ng, Jordan and Weiss, helps with unequal cluster sizes).
5. Run k-means on the rows of $U$; point $i$ gets the cluster of row $i$.

~~~dg-flow Spectral clustering pipeline.
Data points -(kNN graph with Gaussian weights)-> Similarity graph -(normalised Laplacian)-> L_sym -(k smallest eigenvectors)-> *Spectral embedding -(row-normalise, k-means)-> Cluster labels
~~~

## Building the similarity graph

This is the step that decides the result.

- $k$-nearest-neighbour graph: connect each point to its nearest neighbours, then symmetrise. Robust to varying density; a common default is $k$ around $\log n$ to a few tens.
- $\varepsilon$-neighbourhood graph: connect points closer than $\varepsilon$. Sensitive to density differences.
- Gaussian weights: $w_{ij} = \exp(-\|x_i - x_j\|^2/2\sigma^2)$. If $\sigma$ is too large, everything is similar and structure disappears; too small and the graph falls apart into many components. A common choice sets $\sigma$ from the distance to the $k$-th neighbour, possibly per point.

Feature scaling matters as it does for any distance-based method.

## Why it works: cuts and relaxations

Splitting a graph into groups $A_1, \dots, A_k$ cuts the edges between groups. Minimising the cut weight alone tends to slice off single points, so normalised versions divide by group size or volume. The normalised cut objective
$\mathrm{Ncut}(A_1,\dots,A_k) = \sum_i \frac{\mathrm{cut}(A_i, \bar{A}_i)}{\mathrm{vol}(A_i)}$
penalises small groups. Minimising it exactly is NP-hard. Writing it with indicator vectors and relaxing them to real values turns it into minimising a Rayleigh quotient of the normalised Laplacian, whose solution is its smallest eigenvectors. k-means on the embedding rounds the real-valued solution back to discrete groups.

If the graph has exactly $k$ disconnected components, the $k$ smallest eigenvalues are all zero and the eigenvectors are indicator vectors (after rotation), so the embedding puts each component at a single point. With weak connections between groups, the embedding is a small perturbation of that ideal, which is why the method is reliable when clusters are well separated in the graph.

<div class="ex">
<div class="lab">Worked example</div>

**segmenting an image**

Treating each pixel of a 100 by 100 image as a node, with edges between nearby pixels weighted by colour similarity, gives a sparse graph of 10,000 nodes. Spectral clustering with $k = 4$ separated a red mug, a white table, a blue wall and a dark shadow, where k-means on colour alone had merged the shadow with the mug's darker side. Spatial edges kept each region connected; colour weights kept the boundaries sharp.

</div>

## Choosing the number of clusters

The eigengap heuristic: compute the first several eigenvalues and choose $k$ where the gap $\lambda_{k+1} - \lambda_k$ is largest relative to the earlier gaps. A clear gap indicates $k$ well-separated groups; no clear gap means the data has no obvious cluster structure at this scale.

## Cost and scale

A dense similarity matrix needs $O(n^2)$ memory, and a dense eigendecomposition $O(n^3)$ time, feasible up to perhaps tens of thousands of points. For larger data, use a sparse $k$NN graph and an iterative eigensolver for the few smallest eigenvectors, or approximate the similarity matrix with the Nystroem method or landmark points.

## Limitations

- Results depend heavily on graph construction; try several neighbourhood sizes and check stability.
- Clusters of very different densities are hard for a single kernel width.
- New points need the graph rebuilt or an out-of-sample approximation.

<div class="take">
<div class="lab">Key takeaway</div>

spectral clustering embeds points with the smallest eigenvectors of a normalised Laplacian built from a similarity graph, then runs k-means; it solves a relaxed normalised cut, finds non-convex clusters, and lives or dies by how the graph is built.

</div>
