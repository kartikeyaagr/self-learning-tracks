import numpy as np

META = dict(title='Logistic regression by gradient descent', tags=['Classification', 'Optimization'], diff='Easy', runtime='numpy',
            lessons=['mlmaths:12.02'], related=['sigmoid-stable', 'log-loss-binary-cross-entropy', 'softmax-regression'])

STATEMENT = r"""
Binary logistic regression models $P(y=1 \mid x) = \sigma(x^\top w + b)$ with $\sigma(z) = 1/(1+e^{-z})$, and is trained by minimising the mean log loss

$$L(w, b) = -\frac{1}{n}\sum_{i=1}^n \Big[y_i \log p_i + (1-y_i)\log(1-p_i)\Big], \qquad p_i = \sigma(x_i^\top w + b).$$

Write `train_logistic_regression(X, y, lr, n_steps)` that runs **full-batch gradient descent** and returns `(w, b)`.

## Procedure (follow exactly)
1. Start from `w = 0` (shape `(d,)`) and `b = 0.0`.
2. Repeat `n_steps` times: compute the gradient of $L$ at the current `(w, b)`, then update both together: `w -= lr * grad_w`, `b -= lr * grad_b`.

## Input / output
- `X`: float array `(n, d)`; `y`: array `(n,)` of 0/1 labels; `lr > 0`; `n_steps >= 0`.
- Return `w` as a float array `(d,)` and `b` as a Python `float`.
- Your sigmoid must not overflow or return `nan` for large $|z|$.
"""

STARTER = '''import numpy as np

def train_logistic_regression(X, y, lr, n_steps):
    """Full-batch gradient descent on mean log loss from w=0, b=0. Return (w, b)."""
    raise NotImplementedError
'''

SOLUTION = '''import numpy as np

def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))              # identical to 1/(1+e^-z), never overflows

def train_logistic_regression(X, y, lr, n_steps):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(n_steps):
        err = sigmoid(X @ w + b) - y                    # dL/dz_i, up to the 1/n factor
        w = w - lr * (X.T @ err) / n
        b = b - lr * err.mean()
    return w, float(b)
'''

HINTS = [
    'The derivative of the log loss with respect to the logit $z_i$ is remarkably simple. Work it out for one sample first.',
    '$\\partial L/\\partial z_i = (p_i - y_i)/n$. Chain it through $z = Xw + b$: $\\nabla_w L = X^\\top (p - y)/n$ and $\\partial L/\\partial b = \\mathrm{mean}(p - y)$.',
    'Compute `err = sigmoid(X @ w + b) - y` once per step and use it for both updates, so `b` is not updated with a gradient computed from the new `w`.',
]

_r = np.random.default_rng(5)
_X = _r.normal(size=(40, 3))
_y = (_X @ np.array([1.5, -2.0, 0.5]) + 0.3 * _r.normal(size=40) > 0).astype(float)
TESTS = [
    (np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0, 0, 1, 1]), 0.5, 1),
    (np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0, 0, 1, 1]), 0.5, 50),
    (np.array([[1.0, 2.0], [2.0, 1.0]]), np.array([1, 0]), 0.1, 0),
    (np.array([[1.0], [1.0], [1.0]]), np.array([1, 1, 1]), 1.0, 20),
    (np.array([[100.0, -50.0], [-80.0, 40.0], [90.0, 10.0], [-70.0, -20.0]]), np.array([1, 0, 1, 0]), 1.0, 30),
    (_X, _y, 0.3, 200),
    (_X[:, :2] * 4.0, _y, 0.05, 100),
]

EXPLANATION = r"""
## Why the gradient is so clean
For one sample, $\ell = -y\log\sigma(z) - (1-y)\log(1-\sigma(z))$ and $\sigma'(z) = \sigma(z)(1-\sigma(z))$. The $\sigma'$ cancels against the denominators and leaves $\partial\ell/\partial z = \sigma(z) - y$. That's the same "prediction minus target" form as linear regression and softmax cross-entropy, because all three are generalised linear models with their canonical link.

## Convexity
$L$ is convex in $(w, b)$: its Hessian is $\frac1n A^\top \mathrm{diag}(p(1-p)) A$, which is positive semidefinite. Gradient descent with a small enough step converges to the global minimum, if one exists.

## Complexity
Each step is two matrix-vector products: $O(nd)$ time per step.

## Common bugs
- `1/(1+np.exp(-z))` overflows (with a warning) for very negative `z`. It still returns 0, but $\log(1-p)$ in a loss function will then blow up. Test 5 has logits in the hundreds.
- Updating `b` using predictions recomputed after `w` changed. That's a different algorithm (closer to coordinate descent).
- Summing instead of averaging the gradient, which silently multiplies the learning rate by $n$.

## Interview follow-ups
- On linearly separable data, what happens to $\lVert w\rVert$ as `n_steps` grows? (It diverges; the loss approaches 0 but never reaches it. Add L2 regularisation to get a finite optimum.)
- Why not train with squared error on $\sigma(z)$? (The loss becomes non-convex and its gradient vanishes when the model is confidently wrong.)
"""
