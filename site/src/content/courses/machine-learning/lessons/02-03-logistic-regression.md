---
id: "02.03"
course: "machine-learning"
module: "02"
slug: "logistic-regression"
title: "Logistic regression"
keyIdeas: ["Logistic regression models the **log-odds** as linear: $\\log\\frac{p}{1-p} = w^\\top x + b$, so $p = \\sigma(w^\\top x + b)$ and a one-unit change in $x_j$ multiplies the odds by $e^{w_j}$.", "Its loss is the Bernoulli negative log-likelihood (binary cross-entropy); in margin form it is $\\log(1 + e^{-y z})$ with $y \\in \\{-1, +1\\}$, which links it to the SVM's hinge loss.", "The gradient $\\frac1n X^\\top(p - y)$ has the same shape as linear regression's; the loss is convex, so gradient descent, Newton (IRLS) or L-BFGS all reach the global optimum.", "On linearly separable data the unregularised optimum does not exist: weights grow without bound. Regularisation (scikit-learn applies L2 with `C=1.0` by default) fixes it.", "The decision boundary $w^\\top x + b = \\log\\frac{t}{1-t}$ is a hyperplane; changing the threshold $t$ slides it without rotating it."]
question: "You train unregularised logistic regression with gradient descent on a dataset that turns out to be perfectly linearly separable. Describe what happens to the training loss, the norm of the weights and the predicted probabilities as iterations increase, why this is a problem in practice, and two ways to fix it."
answer: "Once a separating hyperplane is found, every example has a positive margin $y_i z_i$, and multiplying $w$ by any factor above 1 increases every margin and lowers every $\\log(1 + e^{-y_i z_i})$ term. So the loss keeps falling toward 0 without reaching it, $\\|w\\|$ grows without bound (roughly logarithmically in the number of steps), and predicted probabilities on training points saturate at 0 and 1. The direction of $w$ settles but its scale never does. In practice the optimiser hits its iteration cap and warns about convergence, coefficients and odds ratios are meaningless, and the model is overconfident on new points near the boundary. Fixes: add an L2 (or L1) penalty, which makes the optimum finite and unique; or stop early on validation loss. Collecting more data that overlaps the classes, or merging the rare categories causing separation, also helps."
sources: [{"title": "Cornell CS4780 lecture notes: Logistic Regression", "url": "https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote06.html", "note": "Logistic regression derived as MLE and MAP, with the gradient."}, {"title": "Stanford CS229 main notes", "url": "https://cs229.stanford.edu/main_notes.pdf", "note": "Part on classification and logistic regression, plus Newton's method and the generalised linear model view."}, {"title": "scikit-learn: LogisticRegression", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html", "note": "What C means, which penalty is on by default, and which solver handles which penalty."}]
practice: ["logistic-regression", "log-loss-binary-cross-entropy", "newton-s-method", "sigmoid-stable", "basic-autograd"]
tags: ["log-odds", "cross-entropy"]
tools: ["NumPy", "scikit-learn"]
---

## From scores to probabilities

For a binary label, a linear score $z = w^\top x + b$ ranges over all real numbers, but a probability must lie in $(0, 1)$. Logistic regression connects the two through the sigmoid:

$$p(y = 1 \mid x) = \sigma(z) = \frac{1}{1 + e^{-z}}.$$

Inverting it shows what is really linear: $z = \log\frac{p}{1-p}$, the **log-odds**. Each feature adds a fixed amount to the log-odds, so a one-unit increase in $x_j$ multiplies the odds by $e^{w_j}$. With $w_j = 0.7$ the odds roughly double. That is the interpretation analysts use, and it holds only with the same caveats about collinearity and confounding as in 02.01.

| Probability p | Odds p/(1-p) | Logit z
| --- | --- | --- |
| 0.50 | 1 | 0
| 0.73 | 2.7 | 1
| 0.88 | 7.4 | 2
| 0.95 | 19 | 2.94
| 0.99 | 99 | 4.6

The table is worth remembering: logits of a few units already mean near-certainty, which is why weights do not need to be large to produce confident predictions.

## Cross-entropy from maximum likelihood

Treat each label as a Bernoulli draw with probability $p_i = \sigma(w^\top x_i)$. The negative log-likelihood, averaged, is the binary cross-entropy:

$$L(w) = -\tfrac{1}{n}\sum_i \big[y_i \log p_i + (1 - y_i)\log(1 - p_i)\big].$$

With labels recoded as $y \in \{-1, +1\}$ the same loss reads $\tfrac1n\sum_i \log(1 + e^{-y_i z_i})$, a function of the **margin** $y_i z_i$ only. Large positive margins cost almost nothing; a confidently wrong prediction (margin $-5$) costs about 5. Lesson 02.05 compares this curve with the SVM's hinge loss.

Why not squared error on $\sigma(z)$? It is not convex in $w$, and its gradient contains a factor $\sigma'(z)$ that vanishes when the model is confidently wrong, exactly when it most needs to move. Cross-entropy cancels that factor. In code, always compute the loss from logits, not from rounded probabilities: `np.logaddexp(0, -y * z)` is exact for large $|z|$, while `np.log(sigmoid(z))` returns `-inf` once $\sigma(z)$ rounds to 0.

## Fitting it

The gradient is strikingly simple:

$$\nabla L = \tfrac{1}{n} X^\top (p - y),$$

prediction error times input, the same shape as linear regression. The Hessian $\tfrac1n X^\top S X$ with $S = \mathrm{diag}(p_i(1 - p_i))$ is positive semi-definite, so the loss is convex and any local minimum is global.

```python
import numpy as np
def fit_logreg(X, y, lam=1e-2, lr=0.5, steps=2000):   # y in {0,1}
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ w)))
        w -= lr * (X.T @ (p - y) / len(y) + lam * w)  # L2 keeps w finite
    return w
```

Plain gradient descent works but is slow on badly scaled features. **Newton's method** uses the Hessian; for logistic regression each Newton step is a weighted least-squares solve, hence the name iteratively reweighted least squares (IRLS), and it converges in 5 to 10 iterations for moderate $d$. scikit-learn defaults to L-BFGS, a quasi-Newton method that approximates curvature from recent gradients, and uses SAGA or liblinear for large sparse problems.

**Separable data.** If some hyperplane separates the classes perfectly, scaling $w$ up always lowers the loss, so the minimiser does not exist: weights diverge, probabilities saturate at 0 and 1, and the optimiser runs until it hits its iteration limit. This happens often with many features or rare categories. An L2 penalty makes the optimum finite and unique; scikit-learn's `LogisticRegression` applies one by default, with `C` the *inverse* strength ($C = 1/\lambda$, applied to a summed rather than averaged loss).

## Decision boundaries

Predicting class 1 when $p \ge t$ is the same as $z \ge \log\frac{t}{1-t}$. For $t = 0.5$ the boundary is the hyperplane $w^\top x + b = 0$, with $w$ normal to it. Moving the threshold shifts the hyperplane parallel to itself; it never rotates. The size $\|w\|$ sets how fast probability changes across the boundary: the distance from the boundary to the 0.73 contour is $1/\|w\|$.

~~~dg-flow One prediction: a linear score, squashed to a probability, then cut at a threshold chosen for the costs at hand.
Features x -(w.x + b)-> Logit z -(sigmoid)-> *Probability p -(compare with t)-> Class
~~~

Curved boundaries come from features, not from the model: adding $x_1^2$, $x_2^2$ and $x_1 x_2$ lets a linear boundary in the expanded space become an ellipse in the original one. Lesson 02.06 does this implicitly with kernels. Choosing $t$ is a cost question covered in module 06; whether $p$ can be trusted as a probability is lesson 03.04.

<div class="ex">
<div class="lab">Worked example</div>

**one gradient step by hand**

Weights $w = (2, -1)$, bias $b = -0.5$, one example $x = (1, 0.5)$ with $y = 1$.

Logit $z = 2 - 0.5 - 0.5 = 1.0$, so $p = \sigma(1) = 0.731$ and the loss is $-\ln 0.731 = 0.313$.

Error $p - y = -0.269$, so the gradient is $(-0.269, -0.134)$ for $w$ and $-0.269$ for $b$. A step with learning rate 0.5 gives $w = (2.134, -0.933)$ and $b = -0.366$; the new logit is $1.30$ and $p = 0.79$.

</div>

<div class="take">
<div class="lab">Key takeaway</div>

logistic regression is a linear model of the log-odds fitted by maximum likelihood. The loss is convex and the gradient is error times input; regularise it, or separable data will send the weights to infinity.

</div>
