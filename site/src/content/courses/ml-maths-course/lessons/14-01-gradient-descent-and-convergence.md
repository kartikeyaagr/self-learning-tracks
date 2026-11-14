---
id: "14.01"
course: "ml-maths-course"
module: "14"
slug: "gradient-descent-and-convergence"
title: "Gradient descent and convergence"
keyIdeas: ["Gradient descent repeats $x_{k+1} = x_k - \\eta\\nabla f(x_k)$. On a quadratic, the error along each Hessian eigendirection is multiplied by $1 - \\eta\\lambda_i$ every step, so every behaviour (crawl, zig-zag, blow-up) is read off the number $\\eta\\lambda_i$.", "Stability needs $\\eta < 2/L$, where $L$ is the largest curvature; above it the iterates diverge along the sharpest direction. The safe textbook choice is $\\eta = 1/L$.", "The **condition number** $\\kappa = L/\\mu$ sets the speed: with $\\eta = 1/L$ the flattest direction shrinks by $1 - 1/\\kappa$ per step, so about $\\kappa\\ln(1/\\varepsilon)$ steps are needed. Convex smooth problems converge at $O(1/k)$, strongly convex ones linearly.", "Ill-conditioning, not dimension, is what makes plain gradient descent slow. Standardising features or preconditioning attacks $\\kappa$ directly: in this lesson it turned 10,000 useless steps into 20.", "Stochastic gradients add noise that does not vanish at the optimum, so a constant step only reaches a cloud around it whose size grows with $\\eta$; that is why schedules decay the step (lesson 14.04)."]
question: "You train linear regression with gradient descent on the loss ½·mean squared error, with an intercept and two features: one uniform on [0, 1], the other uniform on [0, 1000]. Any learning rate above about 6e-6 diverges, and at 1e-6 the weight of the first feature and the intercept have barely moved after 10,000 steps. Explain both symptoms with the eigenvalue picture, estimate the numbers, and give the fix."
answer: "The Hessian of ½·MSE is $X^\\top X/n$, the matrix of feature second moments (with a column of ones for the intercept). The large-scale feature has second moment about $1000^2/3 \\approx 3.3\\times10^5$, so the largest eigenvalue is $L \\approx 3.5\\times10^5$ (a simulation with 200 rows gives $3.55\\times10^5$). Stability requires $\\eta < 2/L \\approx 5.6\\times10^{-6}$; above that the error along the sharp direction is multiplied by a number below $-1$ each step and grows geometrically, which is the divergence. The small eigenvalues come from the [0, 1] feature and the intercept, which are also correlated with each other because neither is centred: they are about 0.34 and 0.06. At $\\eta = 10^{-6}$ the slowest direction shrinks by $1 - 10^{-6}\\times0.06$ per step, so after 10,000 steps it has covered well under 1% of the distance; the condition number is about $5.6\\times10^6$, meaning on the order of $\\kappa\\ln(1/\\varepsilon)$, tens of millions of steps, for a few digits of accuracy. The fix is to standardise both features (subtract the mean, divide by the standard deviation). Centring decouples them from the intercept and scaling equalises the curvatures, so all Hessian eigenvalues become close to 1 ($\\kappa \\approx 1.1$ here, or whatever the remaining correlation between the features implies). A step size near 1 is then stable and convergence takes tens of steps. Remember to apply the same mean and scale to new data at prediction time, or convert the weights back to the original units."
sources: [{"title": "Boyd and Vandenberghe: Convex Optimization", "url": "https://web.stanford.edu/~boyd/cvxbook/", "note": "Free book; chapter 9.3 analyses gradient descent and the role of the condition number."}, {"title": "Tibshirani: Convex Optimization lecture notes, gradient descent", "url": "https://www.stat.cmu.edu/~ryantibs/convexopt/lectures/grad-descent.pdf", "note": "Convergence proofs for smooth and strongly convex problems in a few slides."}, {"title": "Bottou, Curtis and Nocedal: Optimization Methods for Large-Scale Machine Learning", "url": "https://arxiv.org/abs/1606.04838", "note": "SGD convergence with constant and diminishing step sizes (section 4)."}, {"title": "Cohen et al.: Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability", "url": "https://arxiv.org/abs/2103.00065", "note": "How real networks behave relative to the 2/L limit."}]
practice: ["learning-rate-stability-and-convergence-bounds", "vanilla-gradient-descent", "batch-vs-stochastic-vs-mini-batch-gd", "gradient-descent-on-non-convex-loss"]
tags: ["condition number", "convergence"]
tools: ["NumPy"]
---

## What you will learn

- What gradient descent does, step by step, and how to run it by hand on a one-weight loss.
- Why a learning rate that is too large makes training explode, and how to compute the exact limit $\eta < 2/L$ from the curvature.
- How the **condition number** $\kappa$ of a problem decides how many steps you need, and why a stretched bowl makes gradient descent zig-zag.
- Why rescaling features can turn ten thousand useless steps into twenty useful ones.
- What changes when gradients are noisy (SGD), and the classic convergence guarantees for convex and non-convex problems.

*Before you start:* you need derivatives and gradients (lesson 09.08 sets gradients to zero to find optima), the Hessian as a description of curvature (07.01), and eigenvalues and eigenvectors of a symmetric matrix (08.03). Nothing else.

## Where this module starts

Lesson 09.08 found optima by setting the gradient to zero and solving. A neural network has millions of parameters and no such formula, so it is trained by an **iterative method**: start somewhere, take a small step that lowers the loss, repeat.

This module explains how those methods behave. This lesson covers plain gradient descent and the one idea behind almost all of its behaviour: curvature. Then come momentum (14.02), Adam and friends (14.03), learning-rate schedules (14.04), Newton-type methods (14.05) and regularisation as optimisation (14.06).

## The idea in one picture

Take the simplest possible loss: a model with one weight $w$ whose loss is

$$f(w) = (w - 3)^2.$$

The best weight is obviously $w^\star = 3$, but pretend we cannot see that and must find it by walking downhill from $w_0 = 0$. The slope there is $f'(0) = 2(0 - 3) = -6$: negative, so the loss goes down if we move right. Gradient descent moves a fixed fraction $\eta$ (the **learning rate**, or step size) of the slope in the downhill direction. Here is what four different learning rates do:

| Learning rate | $w_0$ | $w_1$ | $w_2$ | $w_3$ | $w_4$ | Behaviour
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 0 | 0.6 | 1.08 | 1.464 | 1.771 | creeps towards 3
| 0.5 | 0 | 3 | 3 | 3 | 3 | lands exactly in one step
| 0.9 | 0 | 5.4 | 1.08 | 4.536 | 1.771 | jumps past 3 and back, but closes in
| 1.1 | 0 | 6.6 | -1.32 | 8.184 | -3.221 | jumps further each time: diverges

~~~dg-fig 14.01-oneweight.svg
Left: the same start, four learning rates. Small steps creep in, η = 0.5 lands in one step, η = 0.9 bounces across the minimum while closing in, η = 1.1 bounces further out each time. Right: the loss per step on a log scale. η = 0.1 and η = 0.9 lie exactly on top of each other: both shrink the error by a factor 0.8 per step, one from one side, the other alternating sides.
~~~

In plain words: a small learning rate is safe but slow; one value is exactly right; past it each step overshoots, first with shrinking and then with growing overshoots. This lesson finds *where those boundaries are* for real, many-dimensional problems. The answer depends on how sharply the loss curves.

## Step 1: the update rule

In words: compute the gradient (the direction of steepest increase), and move a small distance in the opposite direction.

$$x_{k+1} = x_k - \eta\,\nabla f(x_k).$$

- $x_k$ is the vector of all parameters after $k$ steps (we write $x$ in this module; in models it is usually called $w$ or $\theta$).
- $\nabla f(x_k)$ is the **gradient** of the loss at $x_k$: one partial derivative per parameter.
- $\eta > 0$ is the **learning rate**.

Number check on the one-weight loss: $\nabla f(w) = 2(w - 3)$. With $\eta = 0.1$ and $w_0 = 0$: $w_1 = 0 - 0.1 \times (-6) = 0.6$. That is the second column of the table.

For a small enough step, $f(x - \eta g) \approx f(x) - \eta\lVert g\rVert^2$ (with $g$ the gradient), so the loss falls. The next steps make "small enough" exact.

<div class="ex">
<div class="lab">Worked example</div>

**three steps by hand**

Loss $f(w) = (w - 3)^2$, gradient $f'(w) = 2(w - 3)$, learning rate $\eta = 0.1$, start $w_0 = 0$.

Step 1: $f'(0) = 2(0 - 3) = -6$. Update $w_1 = 0 - 0.1 \times (-6) = 0.6$. Loss falls from $9$ to $(0.6 - 3)^2 = 5.76$.

Step 2: $f'(0.6) = 2(-2.4) = -4.8$. Update $w_2 = 0.6 + 0.48 = 1.08$. Loss $(1.08 - 3)^2 = 3.6864$.

Step 3: $f'(1.08) = 2(-1.92) = -3.84$. Update $w_3 = 1.08 + 0.384 = 1.464$. Loss $(1.464 - 3)^2 = 2.359$.

Pattern: the error $w - 3$ goes $-3, -2.4, -1.92, -1.536$. Each step multiplies it by exactly $0.8$, and the loss by $0.8^2 = 0.64$. The next step explains where the 0.8 comes from.

</div>

## Step 2: one direction, one curvature

Write the error as $e_k = w_k - w^\star$. For a quadratic $f(w) = \tfrac{\lambda}{2}(w - w^\star)^2$ the gradient is $\lambda e_k$, where $\lambda$ is the **curvature** (the second derivative: how fast the slope changes). The update becomes

$$e_{k+1} = e_k - \eta\lambda e_k = (1 - \eta\lambda)\,e_k.$$

So every step multiplies the error by the same number, $1 - \eta\lambda$. In our example $(w-3)^2 = \tfrac{2}{2}(w - 3)^2$, so $\lambda = 2$, and with $\eta = 0.1$ the factor is $1 - 0.2 = 0.8$, exactly the pattern in the hand example. With $\eta = 0.9$ the factor is $1 - 1.8 = -0.8$: same size, opposite sign, which is why the two curves in the right panel coincide.

~~~dg-fig 14.01-factor.svg
The factor the error is multiplied by each step, as a function of step size times curvature. Below 1 the error shrinks without changing sign; at exactly 1 it vanishes in one step; between 1 and 2 it shrinks but flips side every step; above 2 it grows and the method diverges.
~~~

The figure gives three regimes, all from the single number $\eta\lambda$:

- $0 < \eta\lambda < 1$: smooth approach from one side. Small $\eta\lambda$ means a factor near 1, so progress is slow.
- $1 < \eta\lambda < 2$: overshoot and **oscillate** (zig-zag) while still shrinking.
- $\eta\lambda > 2$: each overshoot is bigger than the last. **Divergence.**

Number check: $\lambda = 2$, so the limit is $\eta < 2/2 = 1$. The table's $\eta = 1.1$ gives a factor $-1.2$: errors $-3, 3.6, -4.32, 5.18$, which matches $w = 0, 6.6, -1.32, 8.18$.

## Step 3: many directions at once

Real losses have many parameters, and they curve by different amounts in different directions. Near a minimum $x^\star$ any smooth loss looks like a quadratic bowl:

$$f(x) \approx f(x^\star) + \tfrac12 (x - x^\star)^\top H\,(x - x^\star),$$

where $H$ is the **Hessian**, the matrix of second derivatives. Its eigenvectors are the bowl's principal axes and its eigenvalues $\lambda_1 \ge \dots \ge \lambda_d > 0$ are the curvatures along them. The gradient is $H e_k$, so the update is $e_{k+1} = (I - \eta H) e_k$. Rotate into the eigenbasis and the matrix becomes diagonal: each direction is its own one-dimensional problem from Step 2.

$$e_{k+1}^{(i)} = (1 - \eta\lambda_i)\,e_k^{(i)}.$$

~~~dg-flow One gradient step on a quadratic, seen in the Hessian's eigenbasis: each direction is a separate one-dimensional problem.
Error e_k -(rotate into eigenbasis)-> Components e_i -(multiply by 1 - eta lambda_i)-> *Contract or diverge -(rotate back)-> Error e_k+1
~~~

One learning rate has to serve every direction at once. Call the largest curvature $L = \lambda_1$ and the smallest $\mu = \lambda_d$.

- Every direction must satisfy $\eta\lambda_i < 2$. The sharpest one is the binding constraint: **$\eta < 2/L$, or the iterates diverge along the sharpest direction.** The usual safe choice is $\eta = 1/L$.
- With $\eta$ that small, the flattest direction shrinks by $1 - \eta\mu$ per step, which is close to 1. It crawls.

~~~dg-fig 14.01-zigzag.svg
Left: the bowl f = ½(x² + 100y²), 60 steps from (1, 1). At η = 0.019 the path bounces across the steep y-direction and is still at x = 0.32; at η = 0.01 the y-error is removed in one step but x has only reached 0.55. Right: rescale y so both curvatures equal 1 and a single step lands on the minimum.
~~~

That is the zig-zag in a narrow valley: the gradient points mostly across the valley, not along it.

## Step 4: the condition number sets the speed

The ratio

$$\kappa = \frac{L}{\mu} = \frac{\text{largest curvature}}{\text{smallest curvature}}$$

is the **condition number**. It measures how stretched the bowl is: $\kappa = 1$ is perfectly round, $\kappa = 1000$ is a long thin trench.

With $\eta = 1/L$ the slow direction shrinks by $1 - \mu/L = 1 - 1/\kappa$ per step. To cut the error by a factor $1/\varepsilon$ you need $(1 - 1/\kappa)^k \le \varepsilon$, which is about $k \approx \kappa \ln(1/\varepsilon)$ steps. **The cost grows in proportion to $\kappa$.**

~~~dg-fig 14.01-kappa.svg
Left: distance to the minimum with η = 1/L for condition numbers 1, 10, 100 and 1000. κ = 1 finishes in one step; κ = 1000 has barely moved after 400. Right: steps needed for a millionfold reduction; the exact count (solid) and the rule of thumb κ·ln(10⁶) (dashed) are almost the same line.
~~~

Number check: at $\kappa = 100$, reaching $10^{-6}$ takes $\ln(10^{-6})/\ln(0.99) \approx 1375$ steps; the rule of thumb gives $100 \times 13.8 = 1382$.

The best fixed step, $\eta = 2/(L + \mu)$, makes the two extreme directions shrink at the same rate $\frac{\kappa - 1}{\kappa + 1}$. That roughly halves the step count but still scales with $\kappa$. Momentum (14.02) gets $\sqrt{\kappa}$; Newton's method (14.05) removes $\kappa$ on a quadratic.

<div class="ex">
<div class="lab">Worked example</div>

**a stretched bowl, $f(x, y) = \tfrac12(x^2 + 100y^2)$**

The Hessian is $\mathrm{diag}(1, 100)$: $L = 100$, $\mu = 1$, $\kappa = 100$.

Stability requires $\eta < 2/100 = 0.02$. At $\eta = 0.021$ the $y$-error is multiplied by $1 - 2.1 = -1.1$ every step and blows up.

At $\eta = 0.01 = 1/L$ the $y$-error is multiplied by $1 - 1 = 0$: gone after one step. But $x$ shrinks by only $1 - 0.01 = 0.99$ per step, so cutting it 100-fold takes $\ln(0.01)/\ln(0.99) = 4.605/0.01005 \approx 458$ steps. After 500 steps $x = 0.99^{500} \approx 0.0066$.

The best fixed step $\eta = 2/101 \approx 0.0198$ gives the factor $99/101 \approx 0.980$ in both directions: about $230$ steps for the same 100-fold cut. Twice as fast, same order.

Rescale $y$ by 10 (substitute $v = 10y$): $f = \tfrac12(x^2 + v^2)$, both curvatures are 1, $\kappa = 1$, and one step with $\eta = 1$ lands on the minimum.

</div>

## Step 5: where bad conditioning comes from in ML

For least squares with loss $\tfrac{1}{2n}\lVert Xw - y\rVert^2$, the Hessian is $H = X^\top X/n$, the matrix of feature second moments. Its eigenvalues are the data's spread along its principal axes (08.03). Two everyday causes inflate $\kappa$:

- **Features on different scales.** A feature in $[0, 1000]$ has second moment around $3 \times 10^5$; one in $[0, 1]$ around $0.33$.
- **Strongly correlated features.** Two near-duplicate columns create a direction with tiny variance, so $\mu$ is tiny.

~~~dg-fig 14.01-scaling.svg
The same 200-row regression with one feature in [0, 1] and one in [0, 1000], both run at η = 1/L. Raw features: κ is about 6 million and the loss is still 1.2 above the optimum after 10,000 steps. Standardised features: κ = 1.11 and the loss reaches rounding error in about 20 steps.
~~~

**Standardising** (subtract the mean, divide by the standard deviation) fixes the scales and, by centring, removes the coupling with the intercept; whitening and batch normalisation go further. The general form is **preconditioning**: step with $P^{-1}\nabla f$ for a matrix $P \approx H$, which makes the bowl rounder. Newton's method (14.05) uses $P = H$; Adam (14.03) a diagonal estimate.

## Gradient descent in code

The same experiment as the figure, in NumPy. Note that the Hessian's eigenvalues predict both the divergence threshold and the crawl before you run a single step.

```python
import numpy as np

rng = np.random.default_rng(0)
n = 200
x1 = rng.uniform(0, 1, n)            # feature on [0, 1]
x2 = rng.uniform(0, 1000, n)         # feature on [0, 1000]
y = 1 + 3 * x1 + 0.002 * x2 + rng.normal(0, 0.1, n)

def gd(X, y, eta, steps):
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        grad = X.T @ (X @ w - y) / len(y)    # gradient of 0.5 * mean squared error
        w = w - eta * grad
    return w

X_raw = np.c_[np.ones(n), x1, x2]
lam = np.linalg.eigvalsh(X_raw.T @ X_raw / n)     # Hessian eigenvalues
print(lam)                 # [6.29e-02 3.36e-01 3.55e+05]
print(lam[-1] / lam[0])    # 5.6e+06   condition number
print(2 / lam[-1])         # 5.6e-06   largest stable step

print(gd(X_raw, y, 1e-6, 10_000))              # [0.0063 0.0063 0.0058]  true: about (1, 3, 0.002)
print(np.abs(gd(X_raw, y, 6e-6, 300)).max())   # 5.8e+13  just above 2/L: diverged

Z = (np.c_[x1, x2] - [x1.mean(), x2.mean()]) / [x1.std(), x2.std()]
X_std = np.c_[np.ones(n), Z]
print(np.linalg.eigvalsh(X_std.T @ X_std / n))   # [0.95 1.   1.05]
print(gd(X_std, y, 1.0, 30))                       # [3.666 0.903 0.582]
print(np.linalg.lstsq(X_std, y, rcond=None)[0])    # [3.666 0.903 0.582]  same
```

With raw features, $10^{-6}$ is below the limit, yet after 10,000 steps the intercept and first weight have moved under 1% of the way; $6 \times 10^{-6}$, 7% above $2/L$, explodes. After standardising, a step of 1 converges in 30 iterations. In PyTorch or JAX only the gradient changes (autodiff).

## Step 6: noisy gradients (SGD)

Training on a big dataset uses a **mini-batch gradient**: the gradient on a random handful of examples. It equals the true gradient plus zero-mean noise, $g_k = \nabla f(x_k) + \xi_k$. This is **stochastic gradient descent (SGD)**.

The catch: at the optimum the true gradient is zero but the noise is not, so with a constant step the iterates jiggle in a cloud around $x^\star$. The cloud grows with $\eta$ and the gradient variance, and shrinks with batch size.

~~~dg-fig 14.01-sgd.svg
SGD on a noisy two-weight regression, one example per step, averaged over 20 runs. A constant η = 0.1 stalls at a squared distance of about 0.14; η = 0.02 stalls lower, at about 0.021; a step that decays over time keeps improving and is at about 0.002 by the end.
~~~

To reach the optimum the step must shrink. The classical (Robbins-Monro) conditions are $\sum_k \eta_k = \infty$ (you can still travel any distance) and $\sum_k \eta_k^2 < \infty$ (accumulated noise stays finite); $\eta_k \propto 1/k$ satisfies both. In practice: start large for progress, then decay, as the schedules of 14.04 do.

## Reading the result: is it converging?

Diagnostics, from cheapest to most informative:

- **Plot the loss on a log scale.** Linear convergence is a straight downward line; the slope tells you the factor per step. A line that flattens early means a slow direction (poor conditioning) or a noise floor (SGD).
- **Watch the gradient norm.** $\lVert\nabla f\rVert$ going to zero is the stopping signal for full-batch methods. Stop when it falls below a tolerance, or when the relative loss change per step is tiny.
- **Estimate $L$.** Power iteration (repeatedly multiply a vector by $H$) gives the top eigenvalue, hence $2/L$. For networks, a learning-rate range test (raise $\eta$ until the loss blows up) finds the limit empirically.

## What goes wrong

| Symptom | Cause | Fix
| --- | --- | --- |
| Loss grows, then becomes `inf` or `nan` | $\eta > 2/L$ along the sharpest direction | Lower $\eta$ (halve until stable); check $L$; clip gradients for rare spikes
| Loss falls fast, then crawls for thousands of steps | Large $\kappa$: flat directions barely move | Standardise features, remove near-duplicate columns, precondition, add momentum (14.02)
| Loss bounces up and down but trends down | Sharp direction in the $1 < \eta\lambda < 2$ zone | Slightly smaller $\eta$, or accept it; momentum damps it
| SGD loss plateaus with visible noise | Constant step: noise floor proportional to $\eta$ | Decay the step (14.04) or increase the batch size
| Converges, but to a poor model | Non-convex loss: stuck near a saddle or poor local minimum | Different initialisation, noise from SGD, better architecture

## Going deeper

**The descent lemma.** A function is **$L$-smooth** if its gradient is Lipschitz with constant $L$: $\lVert\nabla f(x) - \nabla f(y)\rVert \le L\lVert x - y\rVert$. For twice-differentiable $f$ this means all Hessian eigenvalues lie in $[-L, L]$. Smoothness gives a quadratic upper bound on $f$, and plugging in one gradient step yields

$$f(x_{k+1}) \le f(x_k) - \eta\left(1 - \tfrac{L\eta}{2}\right)\lVert\nabla f(x_k)\rVert^2.$$

The bracket is positive exactly when $\eta < 2/L$, and the guaranteed decrease is largest at $\eta = 1/L$, where it is $\lVert\nabla f\rVert^2/(2L)$. This is the general, non-quadratic version of Step 2.

**Rates.** With $\eta = 1/L$:

| Problem class | Guarantee after k steps | Steps to accuracy epsilon
| --- | --- | --- |
| Convex, L-smooth | $f(x_k) - f^\star \le \frac{L\lVert x_0 - x^\star\rVert^2}{2k}$ | $O(1/\varepsilon)$
| $\mu$-strongly convex, L-smooth | $f(x_k) - f^\star \le (1 - 1/\kappa)^k\,(f(x_0) - f^\star)$ | $O(\kappa\log(1/\varepsilon))$
| Non-convex, L-smooth | $\min_{j<k}\lVert\nabla f(x_j)\rVert^2 \le \frac{2L(f(x_0) - f^\star)}{k}$ | finds a stationary point, not a minimum

**Strongly convex** means curvature at least $\mu > 0$ everywhere, so the minimum is unique. The non-convex bound comes from summing the descent lemma: total decrease cannot exceed $f(x_0) - f^\star$, so gradients cannot stay large. It says nothing about which stationary point you reach; gradient descent escapes saddles from almost every start, but possibly slowly.

**Without convexity.** The **Polyak-Lojasiewicz condition** $\tfrac12\lVert\nabla f(x)\rVert^2 \ge \mu(f(x) - f^\star)$ already gives the $(1 - 1/\kappa)^k$ rate; rank-deficient least squares satisfies it without being strongly convex.

**Lower bounds.** For any method that only combines past gradients, there are $L$-smooth, $\mu$-strongly convex problems that need on the order of $\sqrt{\kappa}\log(1/\varepsilon)$ steps. Gradient descent's $\kappa$ is therefore not optimal; Nesterov's accelerated method (14.02) matches the $\sqrt{\kappa}$ bound.

**Line search.** If $L$ is unknown, backtracking (Armijo) halves a large trial step until the loss drops by a fixed fraction of $\eta\lVert\nabla f\rVert^2$, recovering the same rates.

**Edge of stability.** In full-batch training of deep networks the sharpness (top Hessian eigenvalue) tends to rise until it sits near $2/\eta$, then hovers there while the loss keeps falling non-monotonically (Cohen et al., 2021). The $2/L$ rule is a limit real training pushes against, not an exact per-step prediction.

## Key terms

| Term | Meaning
| --- | --- |
| Gradient descent | Repeatedly step against the gradient: $x_{k+1} = x_k - \eta\nabla f(x_k)$
| Learning rate $\eta$ | The step size multiplying the gradient
| Curvature | How fast the slope changes; along a Hessian eigenvector it is the eigenvalue
| Smoothness constant $L$ | Largest curvature; the step must satisfy $\eta < 2/L$
| Strong convexity $\mu$ | Smallest curvature; positive $\mu$ means a unique minimum
| Condition number $\kappa$ | $L/\mu$: how stretched the bowl is; steps needed grow in proportion to it
| Linear convergence | Error multiplied by a fixed factor below 1 each step: a straight line on a log plot
| Preconditioning | Reshaping the problem (scaling, $P^{-1}\nabla f$) so the bowl is rounder
| SGD noise floor | The cloud around the optimum where constant-step SGD stalls, size proportional to $\eta$

<div class="take">
<div class="lab">Key takeaway</div>

on a smooth problem the sharpest curvature caps the learning rate at $2/L$ and the flattest curvature then sets the speed, so gradient descent needs roughly $\kappa \ln(1/\varepsilon)$ steps; fix the conditioning (standardise, precondition) before tuning anything else, and decay the step when gradients are noisy.

</div>
