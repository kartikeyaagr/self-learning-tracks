---
id: "16.02"
course: "ml-maths-course"
module: "16"
slug: "the-graph-laplacian-and-spectral-ideas"
title: "The graph Laplacian and spectral ideas"
keyIdeas: ["The Laplacian $L = D - A$ satisfies $x^\\top L x = \\sum_{(i,j)\\in E} w_{ij}(x_i - x_j)^2$, so it measures how much a signal on the nodes varies across edges.", "$L$ is symmetric and positive semi-definite; its smallest eigenvalue is 0 with the constant eigenvector, and the number of zero eigenvalues equals the number of connected components.", "The second-smallest eigenvalue measures how well connected the graph is, and its eigenvector (the Fiedler vector) suggests a good two-way split.", "Laplacian eigenvectors act like frequencies on the graph: small eigenvalues give smooth patterns, large ones give rapidly varying patterns."]
question: "Consider the path graph on three nodes, 1 - 2 - 3, with unit weights. (a) Write $L$. (b) Its eigenvalues are 0, 1 and 3; find an eigenvector for each and describe it as a pattern on the path. (c) Evaluate $x^\\top L x$ for $x = (1, 0, -1)$ and explain the number."
answer: "(a) Degrees are 1, 2, 1, so $L = \\begin{pmatrix} 1 & -1 & 0 \\\\ -1 & 2 & -1 \\\\ 0 & -1 & 1 \\end{pmatrix}$. (b) Eigenvalue 0: $(1,1,1)$, constant across the path. Eigenvalue 1: $(1, 0, -1)$; check $Lx = (1, 0, -1)$. It changes once, slowly, from one end to the other. Eigenvalue 3: $(1, -2, 1)$; check $Lx = (3, -6, 3)$. It alternates sign at every edge, the fastest pattern possible. (c) $x^\\top L x = (1-0)^2 + (0-(-1))^2 = 2$, the sum of squared differences across the two edges. Normalising by $x^\\top x = 2$ gives 1, the eigenvalue, as expected for an eigenvector."
sources: [{"title": "Wikipedia, Laplacian matrix", "url": "https://en.wikipedia.org/wiki/Laplacian_matrix", "note": "Definitions, normalised variants and properties."}, {"title": "von Luxburg, A Tutorial on Spectral Clustering (2007)", "url": "https://arxiv.org/abs/0711.0189", "note": "Sections 2 and 3 cover graph Laplacians and their properties."}, {"title": "Hamilton, Graph Representation Learning (free book)", "url": "https://www.cs.mcgill.ca/~wlh/grl_book/", "note": "Chapter 2 on spectral graph theory."}]
practice: ["eigendecomposition", "rbf-kernel-matrix", "matrix-rank", "spectral-norm-by-power-iteration"]
tags: ["graph Laplacian", "connected components", "Fiedler vector", "normalised Laplacian"]
tools: ["NumPy", "SciPy"]
---

## The Laplacian as a difference operator

For an undirected graph with weights $w_{ij}$, define $L = D - A$. For any vector $x$ that assigns a number to each node,
$x^\top L x = \sum_{(i,j)\in E} w_{ij}(x_i - x_j)^2$.
Each edge contributes the squared difference between its endpoints. A small value means $x$ is smooth over the graph: connected nodes have similar values. This one identity explains most of the Laplacian's uses.

Consequences:

- $x^\top L x \ge 0$ for all $x$, so $L$ is positive semi-definite and its eigenvalues are non-negative.
- The constant vector gives $x^\top L x = 0$, so 0 is always an eigenvalue with eigenvector $\mathbf{1}$.
- If the graph has $c$ connected components, any vector constant on each component gives zero, so 0 has multiplicity $c$.

## Eigenvalues as frequencies

Order the eigenvalues $0 = \lambda_1 \le \lambda_2 \le \dots \le \lambda_n$ with orthonormal eigenvectors $u_1, \dots, u_n$. By the Rayleigh quotient, $u_k$ minimises $x^\top L x / x^\top x$ among vectors orthogonal to the previous ones. Small eigenvalues therefore have smooth eigenvectors, and large eigenvalues have eigenvectors that flip sign across many edges. On a path or ring graph, the eigenvectors are exactly sines and cosines, which is why the Laplacian eigenbasis is called the graph Fourier basis.

Expanding a node signal in this basis, $x = \sum_k \hat{x}_k u_k$, and shrinking the high-frequency coefficients smooths the signal over the graph. Graph neural networks perform a cheap version of this low-pass filtering (lesson 16.06).

## Connectivity and the Fiedler vector

$\lambda_2$, the algebraic connectivity, is zero if and only if the graph is disconnected, and it is small when the graph has a bottleneck: two well-connected groups joined by few edges. Its eigenvector, the Fiedler vector, tends to take positive values on one group and negative values on the other. Splitting nodes by the sign of the Fiedler vector is a classic way to cut a graph in two, and it is the starting point of spectral clustering (lesson 16.03).

~~~dg-flow From a graph to a two-way split.
Graph -(build L = D - A)-> Laplacian -(smallest eigenvectors)-> *Fiedler vector -(sign of each entry)-> Two groups with few edges between them
~~~

<div class="ex">
<div class="lab">Worked example</div>

**two communities and a bridge**

Two groups of five nodes are each fully connected internally and joined by a single edge. The Laplacian's second eigenvalue is small (about 0.30), while the next one is 5, reflecting the bottleneck. The Fiedler vector takes values of about $-0.33$ on one group and $+0.33$ on the other, with the two bridge nodes closer to zero ($\mp 0.23$). Splitting by sign recovers the two groups exactly. Adding a second bridging edge between two other nodes raises $\lambda_2$ to about 0.63, as the graph becomes better connected.

</div>

## Normalised Laplacians

Graphs with very unequal degrees make $L$ dominated by hubs. Two normalised versions fix this:

- Symmetric: $L_{\text{sym}} = I - D^{-1/2} A D^{-1/2}$. Its eigenvalues lie in $[0, 2]$.
- Random-walk: $L_{\text{rw}} = I - D^{-1}A = I - P$, where $P$ is the random-walk transition matrix (lesson 16.04).
They have the same eigenvalues, and their eigenvectors are related by a rescaling with $D^{1/2}$. The largest eigenvalue of $L_{\text{sym}}$ equals 2 exactly when a connected component is bipartite.

## Computing eigenvectors

For large sparse graphs you never form a dense eigendecomposition. Iterative methods (Lanczos, LOBPCG) find a few of the smallest eigenvectors using only sparse matrix-vector products. Power iteration on a shifted matrix finds extreme eigenvectors simply, though slowly when eigenvalues are close.

## Where the Laplacian appears in ML

- Spectral clustering and graph partitioning (lesson 16.03).
- Laplacian regularisation in semi-supervised learning: add $\lambda\, f^\top L f$ to the loss so that predictions $f$ vary little across edges of a similarity graph, spreading labels to unlabelled neighbours.
- Laplacian eigenmaps: use the smallest non-trivial eigenvectors as a low-dimensional embedding that keeps neighbours close.
- Positional encodings for graph transformers, using the first few Laplacian eigenvectors.
- Graph convolution, derived as a low-order polynomial filter of the Laplacian (lesson 16.06).

<div class="take">
<div class="lab">Key takeaway</div>

the Laplacian measures variation across edges; its zero eigenvalues count components, its second eigenvector reveals bottlenecks, and its eigenvectors form a frequency basis that underlies spectral clustering and graph convolutions.

</div>
