---
id: "14.01"
course: "ml-maths-course"
module: "14"
slug: "gradient-descent-and-convergence"
title: "Gradient descent and convergence"
keyIdeas: ["Gradient descent repeats $x_{k+1} = x_k - \\eta\\nabla f(x_k)$; on a quadratic each Hessian eigendirection shrinks by a factor $|1 - \\eta\\lambda_i|$ per step.", "Stability needs $\\eta < 2/L$, where $L$ is the largest curvature; the safe textbook choice is $\\eta = 1/L$.", "Convex smooth problems converge at $O(1/k)$; strongly convex ones converge linearly, roughly $(1 - 1/\\kappa)^k$, so iterations scale with the condition number $\\kappa = L/\\mu$.", "Ill-conditioning, not dimension, is what makes plain gradient descent slow; rescaling features or preconditioning attacks $\\kappa$ directly.", "Stochastic gradients add noise, so a constant step size only reaches a neighbourhood of the optimum whose size scales with $\\eta$; that is why schedules decay it (lesson 14.04)."]
question: "You train linear regression by gradient descent on two features: one in [0, 1], the other in [0, 1000]. Any learning rate above 1e-6 diverges, and at 1e-6 the weight of the first feature barely moves after 10,000 steps. Explain both symptoms with the eigenvalue picture and give the fix."
answer: "The Hessian of mean squared error is $X^\\top X/n$, whose eigenvalues are roughly the second moments of the features: about 0.33 for the [0, 1] feature and about $3\\times10^5$ for the [0, 1000] feature. Stability requires $\\eta < 2/L \\approx 6.7\\times10^{-6}$, so larger rates diverge along the large-scale feature's direction. At $\\eta = 10^{-6}$ the small-scale direction contracts by $1 - 10^{-6}\\cdot0.33$ per step, so after 10,000 steps it has moved only about 0.3% of the way: the condition number is around $10^6$. Standardise both features (zero mean, unit variance) so the curvatures are comparable; $\\kappa$ drops to near 1 (or to whatever the feature correlation implies), a step size near 1 becomes stable, and convergence takes tens of steps. Centring also removes the large coupling with the intercept."
sources: [{"title": "Boyd and Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Free book; chapter 9.3 analyses gradient descent and the role of the condition number."}, {"title": "Tibshirani: Convex Optimization lecture notes, gradient descent", "url": "https://www.stat.cmu.edu/~ryantibs/convexopt/lectures/grad-descent.pdf", "note": "Convergence proofs for smooth and strongly convex problems in a few slides."}, {"title": "Bottou, Curtis and Nocedal: Optimization Methods for Large-Scale Machine Learning", "url": "https://arxiv.org/abs/1606.04838", "note": "SGD convergence with constant and diminishing step sizes (section 4)."}, {"title": "Cohen et al.: Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability", "url": "https://arxiv.org/abs/2103.00065", "note": "How real networks behave relative to the 2/L limit."}]
practice: ["learning-rate-stability-and-convergence-bounds", "vanilla-gradient-descent", "batch-vs-stochastic-vs-mini-batch-gd", "gradient-descent-on-non-convex-loss"]
tags: ["condition number", "convergence"]
tools: ["NumPy"]
---

## Where this module sits

Lesson 09.08 found optima by setting gradients to zero and 07.01 showed how the Hessian describes curvature. Real models have millions of parameters and no closed-form solution, so they are trained by iterative methods. This module explains how those methods behave: why a learning rate diverges, why momentum and Adam help, why schedules exist and when second-order information pays. The Deep Learning course applies them inside networks; here the focus is the maths that predicts their behaviour.

## The update and the quadratic model

Gradient descent takes a step against the gradient:

$$x_{k+1} = x_k - \eta\,\nabla f(x_k).$$

Near a minimum $x^\star$, any smooth function looks like a quadratic $f(x) \approx f(x^\star) + \tfrac12 (x - x^\star)^\top H (x - x^\star)$. Write the error $e_k = x_k - x^\star$ in the eigenbasis of $H$ with eigenvalues $\lambda_1 \ge \dots \ge \lambda_d > 0$. The update becomes $e_{k+1} = (I - \eta H)e_k$, so each eigencomponent evolves independently:

$$e_{k+1}^{(i)} = (1 - \eta\lambda_i)\,e_k^{(i)}.$$

Everything about gradient descent on a quadratic follows from this line:

- Component $i$ shrinks if $|1 - \eta\lambda_i| < 1$, that is $0 < \eta < 2/\lambda_i$. The tightest constraint comes from the largest eigenvalue $L = \lambda_1$: **$\eta < 2/L$ or the iterates diverge** along the sharpest direction.
- Between $1/\lambda_i$ and $2/\lambda_i$ the component overshoots and oscillates in sign while still shrinking: the zig-zag across a narrow valley.
- The flattest direction, $\mu = \lambda_d$, shrinks by $1 - \eta\mu$, which is close to 1 when $\eta$ must be small to keep the sharp direction stable.

~~~dg-flow One gradient step on a quadratic, seen in the Hessian's eigenbasis: each direction is a separate one-dimensional problem.
Error e_k -(rotate into eigenbasis)-> Components e_i -(multiply by 1 - eta lambda_i)-> *Contract or diverge -(rotate back)-> Error e_k+1
~~~

## Condition number sets the speed

The **condition number** $\kappa = L/\mu$ measures how stretched the bowl is. With $\eta = 1/L$, the slow direction contracts by $1 - 1/\kappa$ per step, so reaching accuracy $\varepsilon$ takes about $\kappa\log(1/\varepsilon)$ iterations. The best fixed step, $\eta = 2/(L+\mu)$, balances the fastest and slowest directions and gives the rate $\frac{\kappa-1}{\kappa+1}$; it changes the constant, not the scaling with $\kappa$.

The general results, for $L$-smooth $f$ (gradient Lipschitz with constant $L$) and $\eta = 1/L$:

| Problem class | Guarantee after k steps | Steps to accuracy epsilon
| --- | --- | --- |
| Convex, L-smooth | $f(x_k) - f^\star \le \frac{L\lVert x_0 - x^\star\rVert^2}{2k}$ | $O(1/\varepsilon)$
| $\mu$-strongly convex, L-smooth | $\lVert e_k\rVert \le (1 - 1/\kappa)^k\lVert e_0\rVert$ | $O(\kappa\log(1/\varepsilon))$
| Non-convex, L-smooth | $\min_{j<k}\lVert\nabla f(x_j)\rVert^2 = O(1/k)$ | finds a stationary point, not a minimum

<div class="ex">
<div class="lab">Worked example</div>

**a stretched bowl, $f(x, y) = \tfrac12(x^2 + 100y^2)$**

The Hessian is $\mathrm{diag}(1, 100)$: $L = 100$, $\mu = 1$, $\kappa = 100$.

Stability requires $\eta < 2/100 = 0.02$. At $\eta = 0.021$ the $y$ component is multiplied by $1 - 2.1 = -1.1$ every step and blows up.

At $\eta = 0.01$ the $y$ error is wiped out in one step ($1 - 1 = 0$), but $x$ shrinks by only 0.99 per step: $\ln(0.01)/\ln(0.99) \approx 458$ steps to cut it by 100x.

Rescaling $y$ by 10 (so both curvatures are 1) makes $\kappa = 1$, and a single step with $\eta = 1$ lands on the minimum.

</div>

```
import numpy as np
H = np.diag([1.0, 100.0])
x = np.array([1.0, 1.0])
for k in range(500):
    x = x - 0.01 * (H @ x)          # grad of 0.5 x^T H x is H x
print(x)                             # ~[0.0066, 0.0]: the flat direction lags
```

## Where conditioning comes from in ML

For least squares, $H = X^\top X/n$, so the eigenvalues are the variances of the data along its principal axes (lesson 08.03). Features on wildly different scales (age in years next to income in dollars) or strongly correlated features give large $\kappa$. Standardising inputs, whitening and batch normalisation are, in part, ways to shrink $\kappa$. Preconditioning generalises this: step with $P^{-1}\nabla f$ for a matrix $P$ that approximates $H$. Newton's method (14.05) uses $P = H$; Adam (14.03) uses a diagonal estimate.

## Stochastic gradients

Training uses mini-batch gradients $g_k = \nabla f(x_k) + \xi_k$ with zero-mean noise $\xi_k$. The noise does not vanish at the optimum, so with a constant step size SGD settles into a ball around $x^\star$ whose size grows with $\eta$ and the gradient variance, and shrinks with batch size. Reaching the optimum needs a decreasing step (classical conditions: $\sum\eta_k = \infty$, $\sum\eta_k^2 < \infty$), which motivates the schedules of lesson 14.04. In deep networks the picture has a twist: training often runs at the **edge of stability**, where the sharpest curvature rises until it sits near $2/\eta$ and the loss decreases non-monotonically, so the quadratic analysis is a guide to the limits rather than an exact prediction.

<div class="take">
<div class="lab">Key takeaway</div>

on any smooth problem the largest curvature caps the learning rate at $2/L$ and the smallest curvature then sets the speed, so gradient descent needs on the order of $\kappa$ steps per digit of accuracy; fix the conditioning (scaling, preconditioning) before tuning anything else.

</div>
