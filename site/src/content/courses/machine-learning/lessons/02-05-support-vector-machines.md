---
id: "02.05"
course: "machine-learning"
module: "02"
slug: "support-vector-machines"
title: "Support vector machines"
keyIdeas: ["Among all separating hyperplanes, the SVM picks the one with the widest **margin**; fixing the closest points at $y_i(w^\\top x_i + b) = 1$ makes the margin $1/\\|w\\|$, so maximising it means minimising $\\tfrac12\\|w\\|^2$.", "The soft margin adds slack for violations; it is equivalent to minimising $\\tfrac{\\lambda}{2}\\|w\\|^2 + \\tfrac1n\\sum_i \\max(0, 1 - y_i f(x_i))$, the regularised **hinge loss**, with $\\lambda = 1/(nC)$.", "Hinge loss is exactly zero beyond the margin, so only **support vectors** (points on or inside the margin) determine the solution; logistic loss is never zero, so every point contributes.", "**Pegasos** solves the primal by SGD: step $\\eta_t = 1/(\\lambda t)$, shrink $w$ by $(1 - \\eta_t\\lambda)$ every step, and add $\\eta_t y_i x_i$ only when the margin is below 1.", "In the dual, data enter only through dot products $x_i^\\top x_j$, which is what makes kernels (next lesson) possible."]
question: "After training a soft-margin linear SVM (fixed C), you delete every training point that is not a support vector and retrain with the same C. What happens to the solution and why? Would the same trick work for L2-regularised logistic regression?"
answer: "The solution is unchanged. Points strictly beyond the margin have $\\alpha_i = 0$ and zero hinge loss, so they contribute nothing to the objective's (sub)gradient at the optimum. The original $(w, b)$ is still feasible for the reduced problem and still satisfies its optimality (KKT) conditions, and because the problem is convex, it is still optimal. This holds in the $C$ form; in the $\\lambda/n$ form you would have to rescale $\\lambda$, since $n$ changes. In practice, solvers use this to shrink the working set. It does not work for logistic regression: $\\log(1 + e^{-m})$ is positive for every finite margin, so every point exerts some pull on $w$. Deleting the well-classified points removes that pull, the regulariser then shrinks $w$ more, and the boundary and probabilities shift. Logistic regression has no sparse set of support vectors."
sources: [{"title": "Pegasos: Primal Estimated sub-GrAdient SOlver for SVM", "url": "https://home.ttic.edu/~nati/Publications/PegasosMPB.pdf", "note": "Shalev-Shwartz, Singer, Srebro and Cotter: the algorithm, its step size and the n-independent convergence rate."}, {"title": "Cornell CS4780 lecture notes: Support Vector Machine", "url": "https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote09.html", "note": "Margin derivation, soft margin and the hinge-loss view."}, {"title": "scikit-learn: Support Vector Machines", "url": "https://scikit-learn.org/stable/modules/svm.html", "note": "SVC vs LinearSVC, the C parameter, complexity and the warning about unscaled data."}, {"title": "Stanford CS229 main notes", "url": "https://cs229.stanford.edu/main_notes.pdf", "note": "The SVM chapter: functional vs geometric margin, Lagrange duality and the KKT conditions."}]
practice: ["svm-with-hinge-loss-sgd", "perceptron", "averaged-perceptron", "subgradients-for-non-smooth-functions"]
tags: ["margin", "hinge loss"]
tools: ["NumPy", "scikit-learn"]
---

## Margins

When two classes are linearly separable, infinitely many hyperplanes separate them. Logistic regression without regularisation does not even settle on one (02.03). The support vector machine makes a specific choice: the hyperplane that stays as far as possible from the nearest training points. A wide margin means small perturbations of the inputs do not flip predictions, and margin-based bounds tie a wider margin to better generalisation regardless of the number of features.

For a classifier $f(x) = w^\top x + b$ and labels $y \in \{-1, +1\}$, the **functional margin** of point $i$ is $y_i f(x_i)$, positive when correct. Its geometric distance to the hyperplane is $y_i f(x_i)/\|w\|$. Because $w$ and $b$ can be rescaled freely, fix the scale so that the closest points have functional margin exactly 1. The margin is then $1/\|w\|$ on each side, and the hard-margin SVM is

$$\min_{w,b}\; \tfrac12\|w\|^2 \quad \text{s.t.}\quad y_i(w^\top x_i + b) \ge 1 \;\; \forall i.$$

## Soft margin and the hinge loss

Real data overlap, and one mislabelled point should not dictate the boundary. Add a slack $\xi_i \ge 0$ per point, allow $y_i f(x_i) \ge 1 - \xi_i$, and pay for slack:

$$\min_{w,b,\xi}\; \tfrac12\|w\|^2 + C\sum_i \xi_i.$$

At the optimum, $\xi_i = \max(0, 1 - y_i f(x_i))$, so the problem is the unconstrained, regularised **hinge loss** $\tfrac{\lambda}{2}\|w\|^2 + \tfrac1n\sum_i \max(0, 1 - y_i f(x_i))$ with $\lambda = 1/(nC)$. That puts the SVM into the module 01 template: a linear class, a surrogate loss, an L2 penalty. Large $C$ (small $\lambda$) tolerates few violations, giving a narrow margin and higher variance; small $C$ gives a wide margin with more violations.

| Margin m = y f(x) | 0-1 loss | Hinge max(0, 1-m) | Logistic log(1+e^-m)
| --- | --- | --- | --- |
| -1 (wrong) | 1 | 2 | 1.31
| 0 (on boundary) | 1 | 1 | 0.69
| 1 (on margin) | 0 | 0 | 0.31
| 2 (safe) | 0 | 0 | 0.13

Both surrogates are convex upper bounds (up to scale) on the 0-1 loss. The difference is the right tail: hinge loss is exactly zero once a point is past the margin, so those points exert no force on the solution. Logistic loss keeps rewarding larger margins, which is what makes its outputs probabilities. SVM scores are not probabilities; if you need them, calibrate (03.04).

## Pegasos: SGD on the primal

For linear SVMs on large datasets, the simplest effective solver is stochastic subgradient descent on the hinge objective. Pegasos (Shalev-Shwartz et al.) picks one example per step and uses step size $\eta_t = 1/(\lambda t)$:

```python
import numpy as np
def pegasos(X, y, lam=1e-3, T=100_000, seed=0):   # y in {-1,+1}
    rng = np.random.default_rng(seed); w = np.zeros(X.shape[1])
    for t in range(1, T + 1):
        i = rng.integers(len(y)); eta = 1.0 / (lam * t)
        if y[i] * (X[i] @ w) < 1:                    # inside the margin: hinge is active
            w = (1 - eta * lam) * w + eta * y[i] * X[i]
        else:                                        # only the regulariser acts
            w = (1 - eta * lam) * w
    return w
```

Every step shrinks $w$ (the regulariser's gradient); only margin violators add to it. The number of iterations needed for accuracy $\epsilon$ scales like $1/(\lambda\epsilon)$ and does not depend on $n$, which is why it suits very large data. The bias is usually handled by appending a constant feature, which regularises it slightly; with standardised features that rarely matters. Compare the perceptron: it updates only on mistakes (margin $\le 0$), never shrinks, and stops at any separating hyperplane, not the widest one.

## The dual and support vectors

Solving the constrained problem with Lagrange multipliers $\alpha_i$ gives the dual:

$$\max_{\alpha}\; \sum_i \alpha_i - \tfrac12\sum_{i,j}\alpha_i\alpha_j y_i y_j\, x_i^\top x_j \quad \text{s.t.}\quad 0 \le \alpha_i \le C,\; \sum_i \alpha_i y_i = 0,$$

with $w = \sum_i \alpha_i y_i x_i$. Two things to notice. The training data appear only as dot products $x_i^\top x_j$, so replacing them with a kernel $k(x_i, x_j)$ gives a nonlinear SVM without ever forming features (02.06). And the optimality conditions sort points into three groups:

~~~dg-layers Where each training point sits decides its multiplier, and only points with a nonzero multiplier shape the boundary.
Beyond the margin | alpha = 0; no influence, could be deleted
*On the margin | 0 < alpha < C; free support vectors, they pin the hyperplane
Inside the margin or misclassified | alpha = C; bounded support vectors, slack is positive
~~~

Typically only a fraction of points are support vectors, which keeps kernel SVM predictions cheap.

**Practical notes.** Standardise features: the margin is a Euclidean distance, so a feature in large units dominates it. Tune $C$ on a log grid by cross-validation. For linear problems use liblinear (`LinearSVC`) or SGD; libsvm (`SVC`) solves the dual and scales roughly between $n^2$ and $n^3$, so it suits tens of thousands of points, not millions.

<div class="ex">
<div class="lab">Worked example</div>

**two Pegasos steps**

$\lambda = 0.1$, step $t = 10$, so $\eta = 1/(0.1 \cdot 10) = 1$. Current $w = (0.5, -0.2)$.

Example $x = (1, 2)$, $y = +1$: margin $0.5 - 0.4 = 0.1 < 1$, so $w \leftarrow 0.9\,(0.5, -0.2) + (1, 2) = (1.45, 1.82)$. Its margin is now $1.45 + 3.64 = 5.09$.

Next step $t = 11$, $\eta = 0.909$, example $x = (2, 1)$, $y = +1$: margin $2.9 + 1.82 = 4.72 \ge 1$, so only the shrink applies: $w \leftarrow (1 - 0.0909)\,w = (1.32, 1.65)$.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

an SVM is L2-regularised hinge loss on a linear model. The hinge's flat region makes the solution depend only on support vectors, and its dual depends only on dot products, which opens the door to kernels.

</div>
