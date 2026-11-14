---
id: "14.06"
course: "ml-maths-course"
module: "14"
slug: "regularisation-as-optimisation-penalties-constraints-proxima"
title: "Regularisation as optimisation: penalties, constraints, proximal steps"
keyIdeas: ["The penalised problem $\\min f + \\lambda R$ and the constrained problem $\\min f$ subject to $R \\le t$ give the same family of solutions for convex problems; $\\lambda$ is the Lagrange multiplier of the budget $t$, and a large price means a small budget.", "For SGD an L2 penalty is exactly **weight decay**: every step multiplies the weights by $(1 - \\eta\\lambda)$. It also improves conditioning, changing $\\kappa$ from $L/\\mu$ to $(L + \\lambda)/(\\mu + \\lambda)$.", "The L1 ball has corners on the axes, so its solutions often have exact zeros; but (sub)gradient descent only jitters around zero and never lands on it.", "**Proximal gradient** takes a gradient step on the smooth loss and then applies the penalty's prox; for L1 that is soft-thresholding, $\\mathrm{sign}(z)\\max(|z| - \\eta\\lambda, 0)$. This is ISTA, $O(1/k)$; FISTA adds Nesterov momentum for $O(1/k^2)$.", "**Projected gradient** handles hard constraints by stepping and then projecting back onto the feasible set (rescale onto a norm ball, clip for non-negativity); it is proximal gradient with an indicator penalty.", "Early stopping is implicit regularisation: on a quadratic, $k$ gradient steps from zero act like ridge with $\\lambda \\approx 1/(\\eta k)$."]
question: "You fit a lasso model with PyTorch by adding lam * w.abs().sum() to the loss and running Adam. The validation score is fine, but none of the 2,000 weights is exactly zero, although the lasso solution should have only about 50 non-zeros. Explain why and how to get the sparse solution."
answer: "The penalty $\\lambda\\lvert w_i\\rvert$ has a kink at zero. Autograd returns the subgradient $\\lambda\\,\\mathrm{sign}(w_i)$ (0 exactly at 0), so the penalty pushes every weight towards zero with a fixed force, however close it already is. A weight at 0.001 is pushed past zero, then pushed back, and keeps oscillating with an amplitude set by the step size; it practically never lands exactly on 0. Adam makes this worse: it normalises each coordinate's update to roughly the learning rate, so even weights whose gradient is tiny keep moving by about $\\eta$ around zero. The optimiser therefore approaches the lasso objective's *value* (which is why validation looks fine) but not its exact zeros, because exact zeros are a property of the non-smooth optimum that a subgradient method cannot reach.\n\nTo get the sparse solution, treat the L1 term exactly instead of differentiating it. (1) Proximal gradient: take the optimiser step on the smooth data loss only (remove the penalty from the loss), then apply soft-thresholding in place, $w \\leftarrow \\mathrm{sign}(w)\\max(|w| - \\eta\\lambda, 0)$, after every step. With plain SGD and $\\eta \\le 1/L$ this is ISTA; add Nesterov extrapolation for FISTA. Every weight whose gradient-step value lands within $\\eta\\lambda$ of zero becomes exactly zero, so the ~1,950 unsupported weights will be zeroed. (Combining a prox step with Adam's per-coordinate scaling is possible but no longer the standard ISTA, so use SGD for this part.) (2) For a linear model, use a dedicated solver such as scikit-learn's `Lasso`, whose coordinate descent solves each one-dimensional subproblem exactly with a soft-threshold. Check the result by counting `(w == 0).sum()` and by the optimality condition: every zero weight should have a loss gradient no larger than $\\lambda$ in size. A pragmatic fallback is to threshold small weights and refit on the selected features, but the proximal step is the principled fix."
sources: [{"title": "Parikh and Boyd: Proximal Algorithms", "url": "https://web.stanford.edu/~boyd/papers/prox_algs.html", "note": "Proximal operators, their interpretations and the proximal gradient method."}, {"title": "Tibshirani: Convex Optimization lecture notes, proximal gradient descent", "url": "https://www.stat.cmu.edu/~ryantibs/convexopt/lectures/prox-grad.pdf", "note": "ISTA for the lasso, convergence rates and acceleration."}, {"title": "Goodfellow, Bengio and Courville: Deep Learning, chapter 7", "url": "https://www.deeplearningbook.org/contents/regularization.html", "note": "Norm penalties as constrained optimisation and the early stopping and L2 equivalence (7.8)."}, {"title": "Hastie, Tibshirani and Friedman: The Elements of Statistical Learning", "url": "https://hastie.su.domains/ElemStatLearn/", "note": "Free book; section 3.4 compares ridge and lasso geometry."}]
practice: ["lasso-regression-using-ista", "lagrange-multipliers-regularization-as-constrain", "subgradients-for-non-smooth-functions", "early-stopping"]
tags: ["proximal gradient", "weight decay", "sparsity"]
tools: ["NumPy"]
---

## What you will learn

- Why adding a penalty and imposing a constraint are two descriptions of the same problem, and what the Lagrange multiplier $\lambda$ means in that picture.
- Why an L2 penalty is the same as "weight decay" for SGD, and how it makes a badly conditioned problem easier to optimise.
- Why the L1 penalty produces exact zeros, and why plain (sub)gradient descent cannot find them.
- How the **proximal gradient** method (ISTA, and its accelerated version FISTA) solves L1 problems with one extra line of code: soft-thresholding.
- How **projected gradient** handles hard constraints, and why stopping training early behaves like ridge regression.

*Before you start:* you need gradient descent, step sizes and the condition number $\kappa$ (14.01), Nesterov momentum (14.02), the AdamW story of decoupled weight decay (14.03), and Lagrange multipliers for constrained problems (09.08).

## From fitting to fitting with a preference

Lessons 14.01 to 14.05, ending with the second-order methods of 14.05, asked how to reach the minimum of a loss $f(\theta)$ quickly. This last lesson asks *which* minimum you want. **Regularisation** expresses a preference among the many fits a model could make: small weights, few non-zero weights, or weights inside a budget. The Machine Learning course treats this statistically; here we take the optimiser's view, where each kind of preference needs a matching tool.

## The idea in one picture

Take a model with two weights, $w_1$ and $w_2$. Without any preference the loss is smallest at $(2.0, 0.9)$, the "unpenalised best fit". Now insist that the weights stay inside a budget of size 1, measured in two different ways:

| Budget | Feasible region | Best point inside it
| --- | --- | --- |
| L2: $w_1^2 + w_2^2 \le 1$ | a disc | $(0.87,\ 0.50)$
| L1: $\lvert w_1\rvert + \lvert w_2\rvert \le 1$ | a diamond | $(1.00,\ 0.00)$

~~~dg-fig 14.06-balls.svg
The loss contours (ellipses) grow outwards from the unpenalised best fit until they first touch the budget region. The round L2 disc is touched at a point where both weights are non-zero; the L1 diamond is touched at its corner, where w₂ is exactly zero.
~~~

Each ellipse is a set of weights with equal loss. Inflate it from the best fit until it just touches the allowed region: that point is the constrained solution. The disc has no corners, so both weights survive, only smaller. The diamond has corners on the axes, and an inflating ellipse often hits a corner first, where one weight is exactly zero. In short: **L2 shrinks, L1 selects**. The rest of the lesson is about how an optimiser reaches those points.

## Step 1: penalties and constraints are the same problem

There are two ways to write "fit the data, but keep $R(\theta)$ small", where $R$ is a measure of size such as $\lVert\theta\rVert_2^2$ or $\lVert\theta\rVert_1$:

$$\min_\theta\ f(\theta) + \lambda R(\theta) \qquad\text{and}\qquad \min_\theta\ f(\theta)\ \text{ subject to }\ R(\theta) \le t.$$

The left is the **penalised** form: $\lambda \ge 0$ is a price you pay per unit of size. The right is the **constrained** form: $t$ is a hard budget. Here $f$ is the training loss and $\theta$ the vector of weights.

For convex $f$ and $R$ the two forms trace out the same family of solutions. If the budget constraint is active at the solution, there is a $\lambda \ge 0$ for which the penalised problem has the same answer, and that $\lambda$ is exactly the **Lagrange multiplier** of the constraint from 09.08. Conversely, a penalised solution $\theta^*$ also solves the constrained problem with $t = R(\theta^*)$. A large price $\lambda$ corresponds to a small budget $t$.

The multiplier measures how much the loss would drop per unit of extra budget.

<div class="ex">
<div class="lab">Worked example</div>

**matching a budget to a price**

One weight, loss $f(w) = \tfrac12 (w - 3)^2$, size $R(w) = \tfrac12 w^2$.

Penalised form: minimise $\tfrac12(w-3)^2 + \tfrac{\lambda}{2} w^2$. Setting the derivative $(w - 3) + \lambda w$ to zero gives $w = 3/(1+\lambda)$.

Constrained form with budget $\lvert w\rvert \le 1$: the unconstrained best $w = 3$ is outside, so the answer sits on the boundary, $w = 1$.

Which price gives the same answer? Solve $3/(1+\lambda) = 1$: $\lambda = 2$.

Check with the Lagrange condition: at $w = 1$ the loss gradient is $1 - 3 = -2$ and the penalty gradient is $\lambda w = 2 \times 1 = 2$. They cancel, so $w = 1$ is a stationary point of the penalised objective with $\lambda = 2$.

</div>

The mapping between $t$ and $\lambda$ depends on the data, so in practice you tune whichever is more convenient.

## Step 2: L2 is weight decay

Take $R(\theta) = \tfrac12 \lVert\theta\rVert^2$. Its gradient is simply $\theta$, so a gradient step with step size $\eta$ on $f + \lambda R$ is

$$\theta \leftarrow \theta - \eta\big(\nabla f(\theta) + \lambda\theta\big) = (1 - \eta\lambda)\,\theta - \eta\,\nabla f(\theta).$$

Every weight is first multiplied by $(1 - \eta\lambda)$, slightly less than 1, then the ordinary gradient step is taken: **weight decay**. For plain SGD the two names describe the identical update. For Adam they do not: Adam rescales the penalty gradient along with the loss gradient, which is why AdamW applies the shrink separately (14.03).

Number check: with $\eta = 0.1$ and $\lambda = 0.01$ the shrink factor is $1 - 0.001 = 0.999$. A weight with no gradient signal shrinks to $0.999^{1000} \approx 0.37$ of its size after 1,000 steps.

L2 also helps the optimiser itself. The penalty adds $\lambda$ to every eigenvalue of the Hessian, so the condition number from 14.01 changes from $\kappa = L/\mu$ to

$$\kappa_\lambda = \frac{L + \lambda}{\mu + \lambda},$$

where $L$ and $\mu$ are the largest and smallest curvatures. With $L = 100$, $\mu = 0.01$ we have $\kappa = 10{,}000$; adding $\lambda = 1$ gives $101/1.01 = 100$, a hundred times better. And when $\mu = 0$ (more features than rows, so some directions are completely flat), any $\lambda > 0$ makes the problem strongly convex with a unique answer.

## Step 3: why L1 breaks gradient descent

The L1 penalty $\lambda\lVert\theta\rVert_1 = \lambda\sum_i \lvert\theta_i\rvert$ is what produces sparsity, but $\lvert\theta_i\rvert$ has a kink at zero. Its slope is $+1$ to the right, $-1$ to the left, and undefined at 0. Any value in $[-1, 1]$ is a valid **subgradient** there.

Subgradient descent uses $\lambda\,\mathrm{sign}(\theta_i)$ as the penalty's "gradient", a push of fixed size $\eta\lambda$ however close $\theta_i$ is to zero. A weight at $0.001$ overshoots past zero, is pushed back, and keeps jittering. You get roughly the lasso's loss value, but not its zeros.

~~~dg-fig 14.06-ista.svg
A 100-feature lasso on 60 rows (λ = 0.1). Left: the subgradient method with a fixed step stalls at a gap of about 0.05, ISTA keeps improving and FISTA is far ahead. Right: a weight whose lasso value is 0; ISTA puts it exactly at 0 after 35 iterations, the subgradient method jitters around 0 for ever.
~~~

In that run scikit-learn's solution has 84 exact zeros among 100 weights. After 500 steps ISTA and FISTA also have 84 exact zeros; the subgradient method has none.

## Step 4: the proximal step

The fix is to split the objective into a smooth part $f$ (the data loss) and a simple non-smooth part $R$ (the penalty), and treat them differently. Take an ordinary gradient step on $f$ only, then "clean up" with an operation that handles $R$ exactly:

$$\theta_{k+1} = \mathrm{prox}_{\eta\lambda R}\big(\theta_k - \eta\nabla f(\theta_k)\big).$$

The **proximal operator** (prox) of a function $h$ is defined as

$$\mathrm{prox}_{h}(z) = \arg\min_u\ \Big( h(u) + \tfrac12\lVert u - z\rVert^2 \Big).$$

In words: find the point $u$ that balances "stay close to the proposal $z$" against "keep $h$ small". For common penalties it has a closed form.

For $h = \tau\lVert\cdot\rVert_1$ the problem splits into one-dimensional problems, one per coordinate, and the answer is **soft-thresholding**:

$$\mathrm{prox}_{\tau\lVert\cdot\rVert_1}(z)_i = \mathrm{sign}(z_i)\,\max\big(\lvert z_i\rvert - \tau,\ 0\big).$$

Every coordinate moves $\tau$ towards zero, and any coordinate already within $\tau$ of zero is set to exactly zero. That band $[-\tau, \tau]$ is a "dead zone", and it is where exact sparsity comes from.

~~~dg-fig 14.06-prox.svg
Three ways to turn the gradient-step point z into a new weight. The L2 prox scales z down but never reaches zero; soft-thresholding subtracts τ from the size and maps the whole band from −τ to τ to exactly zero. The dots are the worked example below.
~~~

The prox of the L2 penalty $\tfrac{\tau}{2}\lVert u\rVert^2$ is $z/(1 + \tau)$, a rescale that never gives zeros. Proximal gradient with the soft-threshold prox is called **ISTA** (iterative shrinkage-thresholding algorithm).

<div class="ex">
<div class="lab">Worked example</div>

**one soft-thresholding step**

Step size $\eta = 0.1$, penalty $\lambda = 2$, so the threshold is $\tau = \eta\lambda = 0.2$.

After the gradient step on $f$, the proposal is $z = (0.5,\ -0.15,\ 0.3,\ -0.9)$.

Coordinate by coordinate: $0.5 \to 0.5 - 0.2 = 0.3$; $\lvert -0.15\rvert = 0.15 \le 0.2$ so $-0.15 \to 0$; $0.3 \to 0.1$; $-0.9 \to -(0.9 - 0.2) = -0.7$.

Result: $(0.3,\ 0,\ 0.1,\ -0.7)$. Every coordinate moved 0.2 towards zero, and the one inside the dead zone became exactly zero.

A subgradient step would instead add $+\eta\lambda = +0.2$ to $-0.15$, giving $+0.05$: non-zero, with its sign flipped, and on the next step it would be pushed back below zero.

</div>

## Step 5: one full ISTA step, and how fast it converges

The step size rule is the same as in 14.01: with $L$ the Lipschitz constant of $\nabla f$ (the largest curvature), $\eta = 1/L$ is safe. For the least-squares loss $f(w) = \frac{1}{2n}\lVert Xw - y\rVert^2$, $L$ is the largest eigenvalue of $X^\top X/n$.

<div class="ex">
<div class="lab">Worked example</div>

**a lasso solved in one ISTA step**

Take $f(w) = \tfrac12\big[(w_1 - 2)^2 + (w_2 - 0.3)^2\big]$ and $\lambda = 0.5$. The Hessian is the identity, so $L = 1$ and $\eta = 1$.

Start at $w = (0, 0)$. The gradient is $w - (2, 0.3) = (-2, -0.3)$, so the gradient step gives $z = (0,0) - 1 \times (-2, -0.3) = (2, 0.3)$.

Soft-threshold with $\tau = \eta\lambda = 0.5$: $2 \to 1.5$, and $0.3 \le 0.5$ so $0.3 \to 0$.

New point $(1.5,\ 0)$. One more step gives $z = (2, 0.3)$ again and the same output, so this is the exact lasso solution. With uncorrelated features, lasso just shrinks each least-squares weight by $\lambda$ and drops those smaller than $\lambda$.

</div>

With $\eta = 1/L$, ISTA's objective gap falls like $O(1/k)$, as for gradient descent on smooth convex problems. **FISTA** adds Nesterov momentum (14.02), taking the gradient-plus-prox step from an extrapolated point, and reaches $O(1/k^2)$. In the ISTA figure, after 50 iterations FISTA's gap was about $5 \times 10^{-4}$ and ISTA's $6 \times 10^{-2}$. These are worst cases: once the non-zero set settles, convergence is often linear (the straight tail of the ISTA curve).

## Proximal gradient in code

ISTA and FISTA fit in a dozen lines of NumPy. Here they are checked against scikit-learn's `Lasso`, which minimises the same objective $\frac{1}{2n}\lVert Xw - y\rVert^2 + \alpha\lVert w\rVert_1$ by coordinate descent.

```python
import numpy as np
from sklearn.linear_model import Lasso

def soft_threshold(z, tau):
    return np.sign(z) * np.maximum(np.abs(z) - tau, 0.0)

def ista(X, y, lam, steps=500, fista=False):
    n = len(y)
    L = np.linalg.norm(X, 2) ** 2 / n            # Lipschitz constant of grad f
    w = np.zeros(X.shape[1]); v = w.copy(); t = 1.0
    for _ in range(steps):
        z = v - X.T @ (X @ v - y) / n / L         # gradient step, eta = 1/L
        w_new = soft_threshold(z, lam / L)        # prox step, threshold eta*lam
        if fista:                                 # Nesterov extrapolation
            t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
            v = w_new + (t - 1) / t_new * (w_new - w); t = t_new
        else:
            v = w_new
        w = w_new
    return w

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 50))
w_true = np.zeros(50); w_true[:5] = [3, -2, 1.5, 1, -0.5]
y = X @ w_true + rng.normal(0, 0.5, 200)

w_ista = ista(X, y, lam=0.1)
w_fista = ista(X, y, lam=0.1, fista=True)
w_sk = Lasso(alpha=0.1, fit_intercept=False).fit(X, y).coef_
print((w_ista != 0).sum(), (w_fista != 0).sum(), (w_sk != 0).sum())   # 6 6 6
print(np.abs(w_ista - w_sk).max())                                     # 1.1e-05
print(np.round(w_ista[:6], 3))   # [ 2.891 -1.911  1.36   0.868 -0.375  0.   ]
```

All three keep the five true signals plus one tiny extra weight (0.003) and set the other 44 to exactly zero. Each kept weight is shrunk (2.89 instead of 3): the price lasso pays for selection.

In PyTorch, `torch.optim.SGD(params, lr=eta, weight_decay=lam)` gives the same update as adding `lam / 2 * (w ** 2).sum()` to the loss (in a 100-step test the two weight vectors agreed to the last bit). For L1 in a neural network, the proximal version is to take the optimiser step and then apply `soft_threshold` to the weights in place, with threshold $\eta\lambda$ (learning rate times penalty).

## Step 6: hard constraints with projected gradient

Sometimes the constraint matters in its own right, not as a stand-in for a penalty: probabilities must lie on the simplex, embedding vectors must have norm at most $r$, rates must be non-negative. **Projected gradient descent** takes an ordinary gradient step and then moves the result back to the nearest feasible point:

$$\theta_{k+1} = \Pi_C\big(\theta_k - \eta\nabla f(\theta_k)\big),$$

where $C$ is the feasible set and $\Pi_C(z)$ is the point of $C$ closest to $z$ (the **Euclidean projection**).

This is proximal gradient again: take $R$ to be the **indicator** of $C$ (zero inside, $+\infty$ outside), and its prox is exactly the projection.

~~~dg-fig 14.06-project.svg
Projected gradient on a unit disc. Each gradient step (arrow) heads towards the unconstrained minimum and leaves the disc; the projection (dotted) pulls it back to the boundary. The iterates slide along the boundary to the best feasible point, about (0.97, 0.23).
~~~

Common projections are cheap:

- L2 ball of radius $r$: rescale, $\theta \leftarrow \theta\cdot\min(1,\ r/\lVert\theta\rVert)$. The point $(3, 4)$ has norm 5, so projecting onto the unit ball gives $(0.6, 0.8)$. Max-norm constraints in neural networks apply this per unit (per row of a weight matrix).
- Non-negativity or a box $[a, b]$: clip each coordinate.

Projection is cheap only for simple sets; for a complicated set, computing $\Pi_C$ is itself a hard problem. Unlike a penalty, a norm cap does nothing to weights already inside the budget.

## Reading the result: regularisation paths

To see what a regulariser is doing, fit it for a whole range of $\lambda$ values and plot every weight against $\lambda$. This is the **regularisation path**.

~~~dg-fig 14.06-path.svg
Eight features, four of them pure noise. Moving right weakens the penalty. Ridge weights all grow smoothly from zero and none is ever exactly zero. Lasso weights stay at exactly zero until λ drops below a threshold, then enter one at a time; the noise features stay near zero.
~~~

On the lasso side the first weight becomes non-zero at $\lambda \approx 1.73$ and the third at $\lambda \approx 0.75$. Practical checks:

- Count exact zeros (`(w == 0).sum()`). A "lasso" with none was not solved proximally.
- Compare with `sklearn.linear_model.Lasso` on a small problem, minding its $\frac{1}{2n}$ scaling.
- ISTA's objective never increases with $\eta \le 1/L$; an increase means the step is too large.
- Pick $\lambda$ by validation error along the path, never by training error.

## Step 7: early stopping is a regulariser too

You do not need a penalty to prefer small weights. Start gradient descent at $\theta_0 = 0$ and stop before convergence: the weights simply have not travelled far yet.

On a least-squares problem this can be made precise. Let the Hessian have eigenvalues $h_i$ (curvatures of the eigen-directions). After $k$ steps from zero, the component of $\theta$ along direction $i$ has reached a fraction

$$1 - (1 - \eta h_i)^k$$

of its least-squares value. Ridge with penalty $\alpha$ reaches the fraction $h_i/(h_i + \alpha)$ instead. Both are close to 1 for steep, well-determined directions and close to 0 for flat, poorly determined ones, and they line up when $\alpha \approx 1/(\eta k)$. The number of steps plays the role of $1/\lambda$.

~~~dg-fig 14.06-earlystop.svg
Left: the fraction of each direction's least-squares value reached after k = 50 steps with η = 0.1, against the ridge filter with α = 1/(ηk) = 0.2. Right: on a five-feature regression, the weights along the gradient-descent path (solid) roughly follow the ridge solutions with α = 1/(ηk) (dashed).
~~~

Number check with $\eta = 0.1$, $k = 50$, $\alpha = 0.2$: at $h = 2$ gradient descent reaches 1.00 and ridge 0.91; at $h = 0.02$, 0.095 and 0.091; at $h = 0.2$, 0.64 and 0.50. Close, not identical. So learning rate, schedule length and weight decay interact in deep networks: all control how far the weights travel from initialisation.

## What goes wrong

| Symptom | Likely cause | Fix
| --- | --- | --- |
| Lasso fitted with Adam or SGD has no exact zeros | subgradient of $\lvert w\rvert$ makes weights jitter around 0 | proximal step (soft-threshold after each update), coordinate descent, or `sklearn.Lasso`
| Weight decay seems to do nothing under Adam | penalty gradient rescaled by Adam's per-coordinate step | use AdamW (decoupled decay, 14.03)
| ISTA objective goes up | step size above $1/L$ | compute $L$ from the largest singular value, or use backtracking
| Same $\lambda$, different results in two libraries | averaged versus summed loss, $\tfrac12$ or not | rescale $\lambda$ by $n$ or 2
| Large-scale features are punished more than others | penalty applied to raw, unscaled features | standardise features before L1 or L2
| Bias term shrunk towards zero | intercept included in the penalty | exclude biases (and often normalisation parameters) from decay

## Going deeper

**Which proxes are cheap.** Closed forms exist for L1, L2, the group lasso $\sum_g \lVert\theta_g\rVert_2$ (shrink each group's norm, zeroing whole groups), elastic net (soft-threshold then rescale), box and ball indicators, and the nuclear norm of a matrix (soft-threshold its singular values, giving low rank). Overlapping penalties lose the closed form; methods such as ADMM split them further.

**Why ISTA works.** The update minimises a local model: the linearisation of $f$ at $\theta_k$, plus $\frac{1}{2\eta}\lVert\theta - \theta_k\rVert^2$, plus the exact $\lambda R(\theta)$. With $\eta \le 1/L$ that model lies above the objective and touches it at $\theta_k$, so minimising it cannot increase the objective: the descent argument of 14.01, with the non-smooth part kept exact.

**Fixed points are optimal.** $\theta^*$ satisfies $\theta^* = \mathrm{prox}_{\eta\lambda R}(\theta^* - \eta\nabla f(\theta^*))$ exactly when $0 \in \nabla f(\theta^*) + \lambda\,\partial R(\theta^*)$, the optimality condition with subgradients. For the lasso this reads: a weight is zero only if the loss gradient on it is at most $\lambda$ in size. That gives a quick check: weights with $\lvert\nabla_i f\rvert < \lambda$ at the solution must be zero.

**Coordinate descent and minimum norm.** scikit-learn's `Lasso` cycles through coordinates, solving each one-dimensional lasso exactly (again a soft-threshold), with warm starts along the path. And on under-determined least squares, gradient descent from zero converges to the minimum-norm solution: ridge with $\alpha \to 0$.

## Key terms

| Term | Meaning
| --- | --- |
| Penalised form | Minimise loss plus $\lambda$ times a size measure $R(\theta)$
| Constrained form | Minimise loss subject to $R(\theta) \le t$; same solutions as the penalised form for convex problems
| Weight decay | Multiplying weights by $(1 - \eta\lambda)$ each step; equals an L2 penalty for plain SGD
| Subgradient | Any slope that stays below a convex function at a kink; for $\lvert w\rvert$ at 0, any value in $[-1, 1]$
| Proximal operator | $\mathrm{prox}_h(z) = \arg\min_u\, h(u) + \tfrac12\lVert u - z\rVert^2$: the best compromise between staying near $z$ and keeping $h$ small
| Soft-thresholding | Prox of the L1 norm: move each coordinate $\tau$ towards zero, zeroing those within $\tau$
| ISTA / FISTA | Proximal gradient with soft-thresholding; FISTA adds Nesterov momentum, $O(1/k^2)$ instead of $O(1/k)$
| Projected gradient | Gradient step followed by projection onto the feasible set; the prox of an indicator function
| Regularisation path | Every fitted weight plotted against the penalty strength
| Implicit regularisation | Regularisation that comes from the optimiser (early stopping, initialisation) rather than the objective

<div class="take">
<div class="lab">Key takeaway</div>

match the optimiser to the regulariser: weight decay for L2, a proximal step (soft-thresholding, as in ISTA and FISTA) for L1, projection for hard constraints, and treat training length itself as a regularisation knob worth roughly $\lambda \approx 1/(\eta k)$.

</div>
