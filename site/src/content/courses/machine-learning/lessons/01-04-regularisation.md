---
id: "01.04"
course: "machine-learning"
module: "01"
slug: "regularisation"
title: "Regularisation"
keyIdeas: ["Regularisation adds a preference for simple solutions: minimise $\\hat R_S(w) + \\lambda\\,\\Omega(w)$, trading a little bias for lower variance.", "**L2** ($\\|w\\|_2^2$) shrinks all weights smoothly; **L1** ($\\|w\\|_1$) drives many weights exactly to zero.", "A penalty is equivalent to a hard constraint $\\Omega(w) \\le t$ and to a prior in MAP estimation: Gaussian for L2, Laplace for L1.", "**Early stopping** limits how far gradient descent travels from its start, and for quadratic losses behaves much like L2."]
question: "You fit L1-regularised logistic regression on 500 features, two of which are near-duplicates (correlation 0.99) and both truly predictive. What will the lasso likely do with them, why, and how does switching to L2 or elastic net change the outcome?"
answer: "The lasso will usually keep one of the pair and set the other to exactly zero, and which one survives can flip between bootstrap samples. Because the two features carry almost the same information, the loss is nearly unchanged if weight moves from one to the other, while the L1 penalty $|w_1| + |w_2|$ is the same for any split with the same total, so the corner solution (one weight zero) is as cheap as any split and the constant L1 push removes the weaker one. L2 penalises $w_1^2 + w_2^2$, which for a fixed total is smallest when the weight is shared equally, so ridge gives both about half the weight and is stable. Elastic net mixes the two: the L2 part encourages grouping (both kept with similar weights) while the L1 part still zeros out the many irrelevant features."
sources: [{"title": "Deep Learning (Goodfellow, Bengio, Courville), chapter 7: Regularization", "url": "https://www.deeplearningbook.org/contents/regularization.html", "note": "Sections 7.1 and 7.8: L2/L1 analysis and the early-stopping-as-L2 derivation."}, {"title": "Dive into Deep Learning: Weight decay", "url": "https://d2l.ai/chapter_linear-regression/weight-decay.html", "note": "A short runnable demonstration of L2 on a high-dimensional regression."}, {"title": "The Elements of Statistical Learning (Hastie, Tibshirani, Friedman)", "url": "https://hastie.su.domains/ElemStatLearn/", "note": "Free PDF. Section 3.4 on shrinkage methods and the constraint geometry."}]
practice: ["ridge-regression", "lasso-regression", "early-stopping", "lagrange-multipliers-regularization-as-constrain", "adamw-decoupled-weight-decay"]
tags: ["regularisation", "early stopping"]
tools: ["NumPy"]
---

## The idea

Lesson 01.02 showed that a flexible class has high variance: it fits noise. Regularisation keeps the flexible class but adds a preference for "simple" members of it. The regularised objective is

$$\min_w \; \hat R_S(w) + \lambda\,\Omega(w)$$

where $\Omega$ measures complexity and $\lambda \ge 0$ sets the exchange rate. $\lambda = 0$ is plain ERM; $\lambda \to \infty$ forces $\Omega(w) \to 0$, usually $w = 0$, a constant model. Somewhere between is a model with a bit more bias and much less variance, and validation (lesson 01.03) tells you where.

Why does a small norm mean simple? For a linear model $w^\top x$, a small $\|w\|$ means the output changes slowly as the input changes: the function is smooth and cannot chase individual noisy points. Large weights that cancel each other out are the typical signature of overfitting, especially with correlated features.

## L2 and L1

The two standard penalties behave differently:

| Property | L2 (ridge, weight decay) | L1 (lasso) |
| --- | --- | --- |
| Penalty | $\lambda\sum_j w_j^2$ | $\lambda\sum_j |w_j|$ |
| Effect on weights | shrinks all toward 0, rarely exactly 0 | sets many exactly to 0 |
| Gradient of penalty | $2\lambda w_j$, proportional to size | $\lambda\,\text{sign}(w_j)$, constant push |
| Correlated features | spreads weight across them | tends to pick one arbitrarily |
| Closed form (least squares) | yes | no; coordinate descent or proximal methods |
| Prior (MAP view) | Gaussian | Laplace |

The gradient row explains the sparsity. Under L2 the pull toward zero weakens as a weight gets small, so weights approach zero but stop short. Under L1 the pull is constant, so a weight whose data gradient is weaker than $\lambda$ gets pushed all the way to zero and stays there. Module 02 (lesson 02.02) works through both for linear regression, including how to pick $\lambda$.

~~~dg-layers Three views of the same regulariser. Each is useful for a different question.
*Penalty | minimise loss + lambda times norm; what you implement
Constraint | minimise loss subject to norm at most t; explains the geometry (corners of the L1 ball give zeros)
Prior | MAP estimate with a Gaussian or Laplace prior on weights; explains what lambda means
~~~

## Three equivalent views

- **Constraint**: minimising $\hat R + \lambda\|w\|^2$ gives the same solution as minimising $\hat R$ subject to $\|w\|^2 \le t$ for some $t$ that depends on $\lambda$ (a Lagrange multiplier argument). The L1 ball is a diamond with corners on the axes; the loss contours tend to touch it at a corner, where some coordinates are zero.
- **Bayesian**: with Gaussian noise and prior $w \sim \mathcal{N}(0, \tau^2 I)$, the MAP estimate is ridge with $\lambda = \sigma^2/\tau^2$. A tight prior (small $\tau$) means strong regularisation. A Laplace prior gives the lasso.
- **Stability**: regularised solutions change less when one training point changes, which is exactly lower variance.

## Early stopping and other implicit regularisers

Gradient descent started at $w = 0$ moves outward step by step. Stopping after $T$ steps keeps $w$ within a region around the origin whose size grows with $T\eta$ (steps times learning rate). For a quadratic loss this is close to L2 regularisation with $\lambda \approx 1/(T\eta)$; Goodfellow et al. (Deep Learning, chapter 7) give the derivation. In practice you monitor validation loss and keep the checkpoint with the best value, stopping once it has not improved for a set number of evaluations (the *patience*).

Other regularisers you will meet later: data augmentation, dropout, label smoothing, ensembling, and simply using a smaller class. They all reduce variance at some cost in bias.

```python
import numpy as np
def ridge(X, y, lam):
    # Closed form; standardise X first and leave the intercept unpenalised (centre y).
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
```

## Practical rules

1. **Standardise features** before L1 or L2. The penalty treats all weights equally, so a feature measured in millimetres gets penalised far more than the same feature in metres.
2. **Don't penalise the intercept.** Shifting all targets by a constant should shift the prediction, not fight the penalty.
3. **Search $\lambda$ on a log grid**, e.g. $10^{-4}$ to $10^{2}$, because its effect is multiplicative.
4. **Weight decay in adaptive optimisers** is not identical to an L2 term in the loss; AdamW decouples them for that reason (covered in the Deep Learning course).

<div class="ex">
<div class="lab">Worked example</div>

**twelve noisy points, a degree-9 polynomial**

Unregularised least squares fits all 12 points with coefficients in the thousands, alternating sign; validation MSE is 4.8.

With ridge at $\lambda = 10^{-3}$ on standardised polynomial features, coefficients drop below 5 in magnitude, the curve is smooth, and validation MSE is 0.31. At $\lambda = 10$ the curve is almost flat and validation MSE rises to 1.2: too much bias. The best $\lambda$ sits in the middle of the grid, where it should.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

regularisation buys lower variance with a little bias by penalising weight size. L2 shrinks, L1 selects, early stopping limits how far the optimiser travels, and lambda is always chosen on validation data.

</div>
