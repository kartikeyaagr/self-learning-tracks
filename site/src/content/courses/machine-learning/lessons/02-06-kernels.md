---
id: "02.06"
course: "machine-learning"
module: "02"
slug: "kernels"
title: "Kernels"
keyIdeas: ["A **kernel** $k(x, x') = \\phi(x)^\\top\\phi(x')$ computes an inner product in a feature space without building $\\phi$; any algorithm that touches data only through inner products can be kernelised.", "By the representer theorem, the regularised solution is $f(x) = \\sum_i \\alpha_i k(x_i, x)$: one coefficient per training point, not per feature.", "The RBF kernel $e^{-\\gamma\\|x - x'\\|^2}$ corresponds to an infinite-dimensional feature space; $\\gamma$ sets the bump width, from near-linear (small) to memorising each point (large).", "Kernel ridge regression solves $\\alpha = (K + \\lambda I)^{-1}y$: $O(n^3)$ time and $O(n^2)$ memory, so 50,000 points already need a 20 GB Gram matrix.", "Past that scale, approximate the kernel with explicit features (random Fourier features, Nyström) and go back to a linear model."]
question: "You fit RBF kernel ridge regression and set gamma to an extremely large value. What does the Gram matrix approach, what do the predictions on training points and on new test points look like, and what does the opposite extreme (gamma near zero) do?"
answer: "As $\\gamma \\to \\infty$, $k(x_i, x_j) = e^{-\\gamma\\|x_i - x_j\\|^2}$ goes to 0 for every distinct pair and stays 1 on the diagonal, so $K \\to I$. Then $\\alpha = (I + \\lambda I)^{-1}y = y/(1+\\lambda)$, and training predictions are $y_i/(1+\\lambda)$: near-memorisation, shrunk only by the ridge term. A new point is far, on that tiny length scale, from every training point, so all its kernel values are about 0 and it predicts about 0 (the mean, if $y$ was centred). That is a lookup table: training error near zero, test error near the variance of $y$. As $\\gamma \\to 0$, every kernel value approaches 1 and $K$ approaches the all-ones matrix, so the model can represent little beyond a constant. To second order the kernel behaves like a low-degree polynomial in $x$, so the fit becomes very smooth and underfits. The useful $\\gamma$ sits between, chosen by cross-validation jointly with $\\lambda$."
sources: [{"title": "Cornell CS4780 lecture notes: Kernels", "url": "https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote13.html", "note": "Kernelising linear algorithms, well-defined kernels and kernel composition rules."}, {"title": "Random Features for Large-Scale Kernel Machines", "url": "https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html", "note": "Rahimi and Recht (NeurIPS 2007): random Fourier features and their approximation guarantee."}, {"title": "scikit-learn: Kernel ridge regression", "url": "https://scikit-learn.org/stable/modules/kernel_ridge.html", "note": "KernelRidge vs SVR, with timing comparisons that show the n-squared and n-cubed costs."}, {"title": "Gaussian Processes for Machine Learning", "url": "https://gaussianprocess.org/gpml/", "note": "Rasmussen and Williams, chapters 2 and 4: the GP view of kernel ridge and a catalogue of kernels. Free PDF."}]
practice: ["rbf-kernel-matrix", "euclidean-distance", "cholesky-decomposition", "svm-with-hinge-loss-sgd"]
tags: ["kernel trick", "RBF kernel"]
tools: ["NumPy", "scikit-learn"]
---

## Feature maps and their cost

Every model in this module is linear in its features. To get a curved boundary you expand the features: $\phi(x)$ adds squares, products, higher powers. The cost explodes. All monomials up to degree $p$ in $d$ variables number $\binom{d+p}{p}$; for $d = 100$ and $p = 3$ that is 176,851 features per example. Most of that work is wasted if the algorithm only ever needs inner products between examples.

## The kernel trick

Look back at the SVM dual (02.05): the data appear only as $x_i^\top x_j$. Ridge regression can be rewritten the same way. If we replace every inner product with a function $k(x, x')$ that equals $\phi(x)^\top\phi(x')$ for some $\phi$, the algorithm runs in feature space without ever computing $\phi$.

The polynomial case shows it concretely. For $x \in \mathbb{R}^2$,

$$(x^\top x')^2 = x_1^2 x_1'^2 + 2x_1x_2x_1'x_2' + x_2^2x_2'^2 = \phi(x)^\top\phi(x'),\quad \phi(x) = (x_1^2, \sqrt2\,x_1x_2, x_2^2).$$

One dot product and one square replace building three features; for degree 3 in 100 dimensions, one dot product and a cube replace 176,851.

Two results make this general:

- **Representer theorem.** For a loss on training predictions plus an L2 penalty $\lambda\|w\|^2$, the minimiser lies in the span of the training points' features: $w = \sum_i \alpha_i\phi(x_i)$, so $f(x) = \sum_i \alpha_i k(x_i, x)$. You learn $n$ coefficients no matter how many features $\phi$ has, even infinitely many.
- **Mercer's condition.** A symmetric function is a valid kernel exactly when every Gram matrix $K_{ij} = k(x_i, x_j)$ it produces is positive semi-definite. Sums, positive multiples and products of kernels are kernels, which lets you build new ones.

~~~dg-flow Two ways to get the same nonlinear model; the kernel route costs depend on n, the explicit route on the number of features.
[Explicit] Input x -(expand)-> Features phi(x) -(dot with w)-> Prediction
[Kernel] Input x -(compare with each x_i)-> *k(x_i, x) for all i -(weight by alpha_i, sum)-> Prediction
~~~

## Common kernels

| Kernel | Formula | Feature space | Main knob
| --- | --- | --- | --- |
| Linear | $x^\top x'$ | the inputs themselves | none
| Polynomial | $(\gamma\,x^\top x' + c)^p$ | all monomials up to degree p | degree p, offset c
| RBF (Gaussian) | $\exp(-\gamma\|x - x'\|^2)$ | infinite-dimensional | width gamma
| Sigmoid | $\tanh(\gamma\,x^\top x' + c)$ | not always PSD | rarely worth it

The RBF kernel is the default nonlinear choice. It measures similarity: 1 for identical points, falling toward 0 with distance on a length scale of about $1/\sqrt{\gamma}$. The prediction is a weighted sum of bumps centred on training points. Small $\gamma$ gives wide bumps and an almost linear function; large $\gamma$ gives narrow bumps that fit each point separately. scikit-learn's default `gamma="scale"` is $1/(d \cdot \mathrm{Var}(X))$, a sensible start on standardised data; tune $\gamma$ and the regularisation strength together on a log grid, because they interact. Kernels are distance-based, so standardise first, exactly as for kNN (03.02).

## Kernel ridge regression

Ridge regression's solution can be rewritten as $w = X^\top(XX^\top + \lambda I)^{-1}y$ (multiply out both forms to check). Here $XX^\top$ is the $n \times n$ matrix of inner products, so with a kernel:

$$\alpha = (K + \lambda I)^{-1}y, \qquad f(x) = \sum_i \alpha_i\, k(x_i, x).$$

```python
import numpy as np
def rbf(A, B, gamma):
    d2 = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-gamma * np.maximum(d2, 0))

def krr_fit(X, y, gamma, lam):
    K = rbf(X, X, gamma)
    return np.linalg.solve(K + lam * np.eye(len(X)), y)   # alpha; Cholesky in practice

def krr_predict(Xtr, alpha, Xnew, gamma):
    return rbf(Xnew, Xtr, gamma) @ alpha
```

The same formula is the posterior mean of a Gaussian process with covariance $k$ and noise variance $\lambda$, which is why the GP literature is a good second reference. Unlike the SVM, kernel ridge's $\alpha$ is dense: every training point participates in every prediction.

The costs are what limit kernel methods. Training is $O(n^3)$ time and $O(n^2)$ memory; prediction is $O(nd)$ per query. At $n = 50{,}000$, $K$ has $2.5 \times 10^9$ entries, 20 GB in float64. Kernel SVMs fare somewhat better thanks to sparsity, but they too slow down past roughly $10^5$ points.

## Scaling: back to explicit features

The way out is to approximate the kernel with a modest number of explicit features and then fit a linear model, which costs $O(nD^2)$ for $D$ features instead of $O(n^3)$.

- **Random Fourier features** (Rahimi and Recht): draw $\Omega \in \mathbb{R}^{D \times d}$ with entries from $\mathcal{N}(0, 2\gamma)$ and $b$ uniform on $[0, 2\pi]$, and set $z(x) = \sqrt{2/D}\cos(\Omega x + b)$. Then $z(x)^\top z(x') \approx e^{-\gamma\|x - x'\|^2}$, with error shrinking like $1/\sqrt{D}$.
- **Nyström**: pick $m$ landmark points, compute the kernel against them, and use a low-rank approximation of $K$.

With $D$ or $m$ in the low thousands, these usually match the exact kernel model closely, and they work with any linear learner from this module.

<div class="ex">
<div class="lab">Worked example</div>

**checking the polynomial kernel**

$x = (1, 2)$ and $x' = (3, 1)$. Kernel: $x^\top x' = 3 + 2 = 5$, squared is $25$.

Features: $\phi(x) = (1, 2\sqrt2, 4)$ and $\phi(x') = (9, 3\sqrt2, 1)$. Their dot product is $9 + 2\sqrt2 \cdot 3\sqrt2 + 4 = 9 + 12 + 4 = 25$. Same answer, without building $\phi$.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

kernels let linear algorithms work in huge or infinite feature spaces by replacing inner products, at a cost that scales with the number of training points. Use them on small and medium data; at scale, approximate them with explicit random or Nyström features.

</div>
