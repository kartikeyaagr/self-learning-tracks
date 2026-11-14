---
id: "16.01"
course: "ml-maths-course"
module: "16"
slug: "graphs-and-their-matrices"
title: "Graphs and their matrices"
keyIdeas: ["A graph $G = (V, E)$ can be stored as an adjacency list, an edge list or an adjacency matrix $A$; real graphs are sparse, so lists and sparse matrices are the practical choices.", "The degree matrix $D$ holds node degrees on its diagonal; $D$, $A$ and their combinations give the matrices that the rest of the module uses.", "Powers of the adjacency matrix count walks: $(A^k)_{ij}$ is the number of walks of length $k$ from $i$ to $j$.", "Many ML datasets are graphs in disguise: user-item interactions form a bipartite graph whose biadjacency matrix is the interaction matrix."]
question: "A social graph has 10 million users and 2 billion friendship pairs (undirected). (a) How much memory would a dense adjacency matrix take at one bit per entry, and how much does an edge list take with 4-byte node IDs storing each undirected edge once? (b) What is the average degree? (c) How would you count the triangles in a small graph using its adjacency matrix, and why is that impractical here?"
answer: "(a) A dense matrix has $10^{14}$ entries; at one bit each that is $1.25\\times10^{13}$ bytes, about 12.5 TB. An edge list with two 4-byte IDs per edge takes $2\\times10^9 \\times 8 = 16$ GB. (b) Each edge adds 1 to two degrees, so the average degree is $2|E|/|V| = 4\\times10^9/10^7 = 400$. (c) The number of triangles is $\\mathrm{tr}(A^3)/6$: $(A^3)_{ii}$ counts closed walks of length 3 from $i$, and each triangle is counted 6 times (3 starting nodes, 2 directions). Forming $A^3$ densely here is infeasible; instead, for each edge, intersect the sorted neighbour lists of its endpoints, which uses only the sparse structure."
sources: [{"title": "Wikipedia, Adjacency matrix", "url": "https://en.wikipedia.org/wiki/Adjacency_matrix", "note": "Definitions and the walk-counting property."}, {"title": "Wikipedia, Sparse matrix", "url": "https://en.wikipedia.org/wiki/Sparse_matrix", "note": "Storage formats such as COO and CSR."}, {"title": "Hamilton, Graph Representation Learning (free book)", "url": "https://www.cs.mcgill.ca/~wlh/grl_book/", "note": "Chapter 2 covers graph statistics and matrix views."}]
practice: ["matrix-multiplication", "matrix-trace", "item-item-cf-scores", "matrix-vector-multiplication"]
tags: ["adjacency matrix", "adjacency list", "degree", "sparse formats"]
tools: ["NumPy", "SciPy"]
---

## Where this module sits

Linear algebra (modules 03 to 08) gave you matrices, eigenvalues and projections. Graphs connect those tools to data made of relationships: social networks, road maps, molecules, citations, user-item interactions, and the computation graphs that backpropagation runs on. This module builds the matrix view of graphs, then uses it for spectral methods, random walks and graph neural networks.

## Basic vocabulary

A graph has nodes $V$ and edges $E$. Edges may be:

- Undirected (friendship) or directed (follows, links, citations).
- Unweighted or weighted (road length, interaction count, similarity).
- Between nodes of one type, or between two types (users and items), in which case the graph is bipartite.

The degree of a node is the number (or total weight) of its edges. In a directed graph, in-degree and out-degree differ. Real graphs usually have heavy-tailed degree distributions: most nodes have few edges and a few hubs have very many.

## Storage

- Adjacency matrix: $A_{ij} = 1$ (or the weight) if there is an edge from $i$ to $j$. Memory $O(|V|^2)$; fast to check one edge; convenient for algebra.
- Adjacency list: for each node, the list of its neighbours. Memory $O(|V| + |E|)$; fast to iterate neighbours.
- Edge list: a list of $(i, j)$ pairs, often with a weight. Graph learning libraries usually store edges this way, as two integer arrays.
- Compressed sparse row (CSR): the adjacency list packed into arrays, giving sparse matrix products and fast neighbour access.

For an undirected graph, $A$ is symmetric. Self-loops put entries on the diagonal.

~~~dg-layers The matrices built from a graph, used throughout this module.
Adjacency A | which nodes connect, and with what weight
Degree D | diagonal matrix of row sums of A
*Laplacian L = D - A | differences across edges; lesson 16.02
Transition P = D^-1 A | random-walk step probabilities; lesson 16.04
Normalised adjacency | the propagation matrix in graph neural networks; lesson 16.06
~~~

## What matrix products mean

Multiplying a vector of node values $x$ by $A$ sums each node's neighbours' values: $(Ax)_i = \sum_j A_{ij} x_j$. Dividing by degree, $D^{-1}Ax$, averages them. This "aggregate over neighbours" operation is the core of graph neural networks.

Powers count walks. $(A^2)_{ij}$ is the number of length-2 walks from $i$ to $j$, which for $i \neq j$ is the number of common neighbours. People-you-may-know systems rely on exactly this quantity (friends of friends), and item-item recommendation uses $R^\top R$, where $R$ is the user-item matrix, to count users who interacted with both items.

<div class="ex">
<div class="lab">Worked example</div>

**common neighbours as a matrix product**

Users A, B and C are friends with D and E; user F is friends only with E. The common-neighbour counts are entries of $A^2$: A and B share two neighbours (D and E), A and F share one (E). A friend recommender that suggests pairs with many common neighbours would rank B above F for A. At scale, the same counts are computed with sparse operations over each user's two-hop neighbourhood rather than by forming $A^2$.

</div>

## Bipartite graphs and interaction matrices

A user-item interaction matrix $R$ (users by items) is the biadjacency matrix of a bipartite graph. The full adjacency matrix is $\begin{pmatrix} 0 & R \\ R^\top & 0 \end{pmatrix}$. Thinking of recommendation as a graph problem suggests graph methods: random walks from a user through items to other items (lesson 16.04), or graph neural networks over the bipartite graph (lesson 16.06).

## Practical notes

- Store graphs sparsely; a dense matrix is only reasonable up to tens of thousands of nodes.
- Keep a mapping between external IDs and contiguous integer indices; most libraries need the latter.
- Decide early whether edges are directed, and whether to add reverse edges; many algorithms assume symmetry.
- Hubs dominate neighbour aggregation and sampling costs; cap or sample neighbours of very high-degree nodes.

## Degree distributions in practice

Average degree hides a lot. In a social graph with average degree 400, the median user may have 150 friends while a few accounts have millions of followers. Algorithms whose cost grows with the square of a node's degree (counting common neighbours, for example) are dominated by those hubs. Check the maximum and high percentiles of degree before choosing an algorithm, and treat extreme hubs separately.

<div class="take">
<div class="lab">Key takeaway</div>

store real graphs as sparse lists or matrices, and use the adjacency, degree and related matrices as tools; matrix-vector products aggregate neighbours and matrix powers count walks.

</div>
