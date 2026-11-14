---
id: "16.04"
course: "ml-maths-course"
module: "16"
slug: "random-walks-markov-chains-and-pagerank"
title: "Random walks, Markov chains and PageRank"
keyIdeas: ["A random walk on a graph moves to a neighbour chosen in proportion to edge weight; its transition matrix is $P = D^{-1}A$ and its state distribution evolves as $\\pi_{t+1} = \\pi_t P$.", "On a connected, non-bipartite undirected graph the walk converges to a stationary distribution proportional to node degree, at a rate set by the spectral gap.", "PageRank is the stationary distribution of a walk that, with probability $1-\\alpha$, jumps to a random node; the jump makes it unique and lets power iteration converge at rate $\\alpha$.", "Short random walks generate training data for node embeddings and power personalised recommendations."]
question: "Three web pages link as follows: A links to B and C, B links to C, C links to A. (a) Write the transition matrix $P$. (b) Find the stationary distribution without damping. (c) With damping $\\alpha = 0.85$, PageRank satisfies $r = \\frac{1-\\alpha}{3}\\mathbf{1}^\\top + \\alpha\\, r P$. The solution is about $(0.388, 0.215, 0.397)$. Explain why C ranks highest and how damping changed the ranking compared with (b)."
answer: "(a) Rows are sources: $P = \\begin{pmatrix} 0 & 0.5 & 0.5 \\\\ 0 & 0 & 1 \\\\ 1 & 0 & 0 \\end{pmatrix}$. (b) Solve $\\pi = \\pi P$: $\\pi_A = \\pi_C$, $\\pi_B = 0.5\\pi_A$, $\\pi_C = 0.5\\pi_A + \\pi_B = \\pi_A$. Normalising, $\\pi = (0.4, 0.2, 0.4)$. (c) C receives links from both A and B, so it collects more of the walk's probability. Without damping, A and C tie because all of C's rank flows to A. Damping sends 15% of each step to a random page, so A no longer receives all of C's rank undiluted, and C edges ahead of A while B rises slightly from the uniform jumps."
sources: [{"title": "Wikipedia, PageRank", "url": "https://en.wikipedia.org/wiki/PageRank", "note": "Definition, damping and computation."}, {"title": "Wikipedia, Markov chain", "url": "https://en.wikipedia.org/wiki/Markov_chain", "note": "Stationary distributions and convergence."}, {"title": "Grover and Leskovec, node2vec: Scalable Feature Learning for Networks (2016)", "url": "https://arxiv.org/abs/1607.00653", "note": "Biased random walks to learn node embeddings."}, {"title": "Wikipedia, Random walk", "url": "https://en.wikipedia.org/wiki/Random_walk", "note": "Background on random walks in general."}]
practice: ["matrix-vector-multiplication", "spectral-norm-by-power-iteration", "word2vec-training-pairs", "eigendecomposition"]
tags: ["random walks", "stationary distribution", "PageRank", "power iteration"]
tools: ["NumPy", "SciPy"]
---

## Random walks on graphs

Start at a node, move to a neighbour chosen with probability proportional to edge weight, and repeat. The transition matrix is $P = D^{-1}A$: row $i$ holds the probabilities of moving from $i$ to each node, and each row sums to 1. If $\pi_t$ is a row vector giving the probability of being at each node at step $t$, then
$\pi_{t+1} = \pi_t P$, so $\pi_t = \pi_0 P^t$.

A random walk is a Markov chain: the next state depends only on the current one. Everything about long-run behaviour follows from the eigenvalues of $P$.

## The stationary distribution

A distribution $\pi$ with $\pi = \pi P$ is stationary: once the walk reaches it, it stays. For a connected undirected graph,
$\pi_i = \frac{d_i}{\sum_j d_j} = \frac{d_i}{2|E|}$
(weighted degrees for weighted graphs). A walk spends time in proportion to how connected a node is.

The walk converges to $\pi$ from any start if the graph is connected and not bipartite. On a bipartite graph the walk alternates between the two sides and never settles; adding a self-loop to each node (a "lazy" walk) fixes this. The speed of convergence depends on the second-largest eigenvalue modulus of $P$: the larger the spectral gap, the faster the walk mixes. $P$ is similar to $I - L_{\text{sym}}$, so this gap is directly related to the Laplacian eigenvalues of lesson 16.02, and graphs with bottlenecks mix slowly.

## PageRank

On the web, links are directed, some pages have no outgoing links, and parts of the graph trap the walk. PageRank fixes all three by teleportation: at each step, with probability $\alpha$ follow a random outgoing link, otherwise jump to a page chosen uniformly (or from a chosen distribution). Pages with no outgoing links jump uniformly. The PageRank vector satisfies
$r = \frac{1-\alpha}{n}\mathbf{1}^\top + \alpha\, r P$,
with $\alpha = 0.85$ the classic choice. The teleport term guarantees a unique solution.

## Computing it: power iteration

Start from the uniform vector and repeat $r \leftarrow \frac{1-\alpha}{n}\mathbf{1}^\top + \alpha\, r P$. The error shrinks by a factor of at least $\alpha$ each step, so 50 iterations reduce it by a factor of about $0.85^{50} \approx 3\times10^{-4}$. Each iteration is one sparse matrix-vector product, which scales to billions of edges on a cluster.

~~~dg-flow Power iteration for PageRank.
Uniform start r0 -(r P: follow links, weight alpha)-> Propagated rank -(+ teleport mass)-> *Updated r -(repeat until change is tiny)-> PageRank scores
~~~

<div class="ex">
<div class="lab">Worked example</div>

**personalised PageRank for recommendations**

Replace the uniform teleport with a jump back to one user's node on a user-item graph. The stationary distribution then concentrates on items near that user in the graph: items liked by users who liked the same things. A recommender that ran short walks with restarts from each user, counted visits to items, and filtered already-seen items produced candidates comparable to a matrix factorisation model, and it updated instantly as new interactions arrived because no retraining was needed.

</div>

## Random walks for embeddings

Treat short random walks as sentences of node IDs and train skip-gram embeddings on them (DeepWalk). Nodes that co-occur on walks get similar vectors, so the embedding reflects graph proximity. node2vec biases the walk with two parameters that trade off breadth-first exploration (capturing structural roles) against depth-first exploration (capturing communities). The resulting embeddings feed classifiers or link predictors, or serve as candidate generators.

## Hitting times and diffusion

Other walk quantities are useful too. The hitting time from $i$ to $j$ (expected steps to reach $j$) and the commute time (there and back) act as graph distances that account for all paths, not just the shortest. Diffusion, $\pi_0 e^{-tL}$ in continuous time, spreads mass smoothly and is used for label propagation and graph-based semi-supervised learning.

## Practical notes

- Normalise directed graphs carefully; nodes without out-links need a rule.
- Hubs attract walks; for recommendations, downweight popular items or normalise visit counts by degree.
- Monte Carlo walks approximate personalised PageRank cheaply when only the top results are needed.

<div class="take">
<div class="lab">Key takeaway</div>

a random walk's transition matrix $P = D^{-1}A$ drives a distribution toward a stationary one (degree-proportional on undirected graphs); PageRank adds teleportation for a unique, fast-converging answer, and short walks underpin node embeddings and graph recommenders.

</div>
