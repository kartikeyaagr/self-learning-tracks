---
id: "14.06"
course: "ml-maths-course"
module: "14"
slug: "regularisation-as-optimisation-penalties-constraints-proxima"
title: "Regularisation as optimisation: penalties, constraints, proximal steps"
keyIdeas: ["Penalised ($\\min f + \\lambda R$) and constrained ($\\min f$ subject to $R \\le t$) problems trace the same family of solutions; $\\lambda$ is the Lagrange multiplier of the constraint.", "For SGD, an L2 penalty is exactly weight decay, a shrink by $(1 - \\eta\\lambda)$ each step; the L1 ball's corners make exact zeros likely, which is why lasso gives sparse solutions.", "Proximal gradient handles non-smooth penalties: take a gradient step on $f$, then apply $\\mathrm{prox}_{\\eta\\lambda R}$; for L1 that is soft-thresholding, and the algorithm is ISTA (FISTA adds Nesterov momentum).", "Projected gradient handles hard constraints by stepping and then projecting back onto the feasible set, such as rescaling onto a norm ball.", "Early stopping is implicit regularisation: on a quadratic, $k$ gradient steps behave like ridge with $\\lambda \\approx 1/(\\eta k)$."]
question: "You fit a lasso model with PyTorch by adding lam * w.abs().sum() to the loss and running Adam. The validation score is fine, but none of the 2,000 weights is exactly zero, although the lasso solution should have only about 50 non-zeros. Explain why and how to get the sparse solution."
answer: "The gradient of $|w_i|$ is $\\pm1$ (the framework picks a subgradient at zero), so each step pushes small weights towards zero but overshoots: a weight at 0.001 is pushed past zero and then pushed back, oscillating with an amplitude set by the step size and never landing exactly on 0. Adam makes it worse by normalising each coordinate's step to about $\\eta$, so tiny weights keep moving by $\\eta$ around zero. The optimiser approximates the lasso objective's value but not its exact zeros. To get the sparse solution, use a method that handles the non-smooth term exactly: proximal gradient (ISTA or FISTA) with soft-thresholding after each gradient step on the smooth loss, or coordinate descent as in scikit-learn's Lasso. Both produce exact zeros. A pragmatic alternative is to threshold the trained weights and refit, but the prox step is the principled fix."
sources: [{"title": "Parikh and Boyd: Proximal Algorithms", "url": "https://web.stanford.edu/~boyd/papers/prox_algs.html", "note": "Proximal operators, their interpretations and the proximal gradient method."}, {"title": "Tibshirani: Convex Optimization lecture notes, proximal gradient descent", "url": "https://www.stat.cmu.edu/~ryantibs/convexopt/lectures/prox-grad.pdf", "note": "ISTA for the lasso, convergence rates and acceleration."}, {"title": "Goodfellow, Bengio and Courville: Deep Learning, chapter 7", "url": "https://www.deeplearningbook.org/contents/regularization.html", "note": "Norm penalties as constrained optimisation and the early stopping and L2 equivalence (7.8)."}, {"title": "Hastie, Tibshirani and Friedman: The Elements of Statistical Learning", "url": "https://hastie.su.domains/ElemStatLearn/", "note": "Free book; section 3.4 compares ridge and lasso geometry."}]
practice: ["lasso-regression-using-ista", "lagrange-multipliers-regularization-as-constrain", "subgradients-for-non-smooth-functions", "early-stopping"]
tags: ["proximal gradient", "weight decay", "sparsity"]
tools: ["NumPy"]
---

## Regularisation is a change to the optimisation problem

The Machine Learning course explains regularisation statistically: penalties trade bias for variance, and ridge, lasso and elastic net are chosen by validation. This lesson takes the optimiser's view. Adding a penalty, imposing a constraint and stopping early all change which point an optimiser reaches, and each needs a matching algorithmic tool. It closes the module by connecting to the constrained optimisation of lesson 09.08 and the weight decay of 14.03.

## Penalties and constraints are two views of one problem

Compare

$$\min_\theta\ f(\theta) + \lambda R(\theta) \qquad\text{and}\qquad \min_\theta\ f(\theta)\ \text{ subject to }\ R(\theta) \le t.$$

For convex $f$ and $R$, every solution of the constrained form with an active constraint solves the penalised form for some $\lambda \ge 0$, and vice versa. That $\lambda$ is the Lagrange multiplier of the constraint (09.08): it measures how much the loss would improve per unit of extra budget $t$. Large $\lambda$ corresponds to small $t$. The mapping between them depends on the data, so in practice you tune whichever is more convenient.

The constrained form gives the geometry. The solution is where the loss contours first touch the constraint set. The L2 ball is round, so it touches at a generic point with all coordinates non-zero: ridge shrinks. The L1 ball is a cross-polytope with corners on the axes, and contours tend to hit a corner or an edge first, where some coordinates are exactly zero: lasso selects.

## L2: weight decay and its gradient

With $R = \tfrac12\lVert\theta\rVert^2$, a gradient step is

$$\theta \leftarrow \theta - \eta(\nabla f + \lambda\theta) = (1 - \eta\lambda)\theta - \eta\nabla f.$$

The penalty is a multiplicative shrink every step, hence "weight decay". For SGD the two names describe the same update; under Adam they differ, which is the AdamW story from 14.03. L2 also improves conditioning: it adds $\lambda$ to every Hessian eigenvalue, so $\kappa$ becomes $(L + \lambda)/(\mu + \lambda)$, and a problem with $\mu = 0$ (more features than rows) becomes strongly convex.

## L1 and the proximal gradient method

The L1 norm is not differentiable at zero, so plain gradient descent, or subgradient descent, oscillates around zero and never produces exact zeros. The **proximal gradient** method splits the objective into a smooth part $f$ and a simple non-smooth part $R$:

$$\theta_{k+1} = \mathrm{prox}_{\eta\lambda R}\big(\theta_k - \eta\nabla f(\theta_k)\big), \qquad \mathrm{prox}_{h}(z) = \arg\min_u \ h(u) + \tfrac12\lVert u - z\rVert^2.$$

The prox finds the point that balances staying close to the gradient step against keeping $R$ small. For $R = \lVert\cdot\rVert_1$ it separates by coordinate into **soft-thresholding**:

$$\mathrm{prox}_{\tau\lVert\cdot\rVert_1}(z)_i = \mathrm{sign}(z_i)\max(|z_i| - \tau, 0).$$

Proximal gradient with this prox is **ISTA**. With $\eta = 1/L$ it converges at $O(1/k)$, like gradient descent on smooth problems; **FISTA** adds Nesterov momentum (14.02) and reaches $O(1/k^2)$.

~~~dg-flow One proximal-gradient (ISTA) step: a plain gradient step on the smooth loss, then the penalty's prox.
theta_k -(gradient step on f)-> z = theta_k - eta grad f -(soft-threshold by eta lambda)-> *theta_k+1, exact zeros
~~~

<div class="ex">
<div class="lab">Worked example</div>

**one soft-thresholding step**

Step size $\eta = 0.1$, penalty $\lambda = 2$, so the threshold is $\tau = \eta\lambda = 0.2$.

After the gradient step, $z = (0.5,\ -0.15,\ 0.3,\ -0.9)$.

Soft-thresholding gives $(0.3,\ 0,\ 0.1,\ -0.7)$: every coordinate moves 0.2 towards zero, and the one already inside $[-0.2, 0.2]$ becomes exactly zero.

A subgradient step would instead move $-0.15$ to $+0.05$ and keep it non-zero, flipping its sign back and forth on later steps.

</div>

```
import numpy as np
def ista(X, y, lam, steps=500):
    L = np.linalg.norm(X, 2) ** 2 / len(y)        # Lipschitz constant of grad f
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        z = w - X.T @ (X @ w - y) / len(y) / L    # gradient step, eta = 1/L
        w = np.sign(z) * np.maximum(np.abs(z) - lam / L, 0.0)
    return w
```

## Hard constraints: projected gradient

When the constraint matters in its own right (a probability simplex, a norm cap on embeddings, non-negative weights), take a gradient step and **project** back onto the feasible set $C$. That is the proximal method with $R$ the indicator of $C$ (zero inside, infinite outside), whose prox is the Euclidean projection. Projecting onto an L2 ball of radius $r$ is a rescale, $\theta \leftarrow \theta\cdot\min(1, r/\lVert\theta\rVert)$; max-norm constraints on network weights apply it per unit. Non-negativity is a clip at zero. Projection is cheap only for simple sets; that is the practical limit of the method.

| Regulariser | Optimisation form | Algorithm | Effect
| --- | --- | --- | --- |
| L2 penalty | smooth term $\tfrac\lambda2\lVert\theta\rVert^2$ | any gradient method; weight decay | shrinks all weights, improves conditioning
| L1 penalty | non-smooth term $\lambda\lVert\theta\rVert_1$ | proximal gradient (ISTA, FISTA), coordinate descent | exact zeros, feature selection
| Norm constraint | $\lVert\theta\rVert \le r$ | projected gradient | hard cap, no shrinkage inside the ball
| Early stopping | stop after $k$ steps | any iterative method | implicit ridge with $\lambda \approx 1/(\eta k)$

## Early stopping as implicit regularisation

Run gradient descent from $\theta_0 = 0$ on a least-squares loss with Hessian eigenvalues $\lambda_i$. After $k$ steps, the component along eigenvector $i$ has reached a fraction $1 - (1 - \eta\lambda_i)^k$ of its least-squares value. Ridge with penalty $\alpha$ reaches the fraction $\lambda_i/(\lambda_i + \alpha)$. Both are near 1 for high-curvature directions and near 0 for low-curvature ones, and they match closely when $\alpha \approx 1/(\eta k)$. Stopping early therefore suppresses the flat, poorly determined directions, just as ridge does; the number of steps plays the role of $1/\lambda$. This is one reason the learning rate, schedule length and weight decay of a deep network interact: all three control how far the optimiser travels from its initialisation.

<div class="take">
<div class="lab">Key takeaway</div>

choose the optimiser to match the regulariser: weight decay for L2, proximal steps (soft-thresholding) for L1, projection for hard constraints, and treat training length itself as a regularisation knob worth roughly $\lambda \approx 1/(\eta k)$.

</div>
